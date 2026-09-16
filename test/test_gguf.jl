# The GGUF reader against files written by the fixture writer: every
# metadata type, custom alignment, every tensor encoding it materializes,
# zero-copy quantized views, and the failure modes.

const G = Wink.GGUF

@testset "GGUF reader" begin
    dir = mktempdir()
    r = InferenceFixtures.SplitMix(0x6606)
    dense = Float32[i + j / 10 for i in 1:3, j in 1:4]
    bf16vals = Float32[1.0, -2.0, 0.5, 3.0]          # exactly representable
    q40 = qblocks(r, :q4_0, 32, 2)
    q41 = qblocks(r, :q4_1, 32, 2)
    q80 = qblocks(r, :q8_0, 32, 2)
    q6k = qblocks(r, :q6_K, 256, 1)
    q40x3 = qblocks(r, :q4_0, 32, 6)                 # 3 experts × 2 columns
    meta = Pair{String, Any}[
        "general.alignment" => UInt32(64),
        "t.u8" => UInt8(200), "t.i8" => Int8(-7), "t.u16" => UInt16(60000),
        "t.i16" => Int16(-300), "t.u32" => UInt32(7), "t.i32" => Int32(-9),
        "t.f32" => 1.5f0, "t.bool" => true, "t.str" => "wink ∘ gguf",
        "t.u64" => UInt64(2)^40, "t.i64" => Int64(-2)^41, "t.f64" => 0.25,
        "t.ints" => Int32[1, -2, 3], "t.strs" => ["a", "bc"],
        "t.bools" => Bool[1, 0, 1], "t.floats" => Float32[0.5, 1.5]]
    tensors = Any[
        ("dense", 0, [3, 4], InferenceFixtures.f32bytes(dense)),
        ("half", 1, [4], collect(reinterpret(UInt8, Float16[1, 2.5, -3, 0]))),
        ("brain", 30, [4], collect(reinterpret(UInt8,
            [UInt16(reinterpret(UInt32, v) >> 16) for v in bf16vals]))),
        ("q40", 2, [32, 2], q40), ("q41", 3, [32, 2], q41),
        ("q80", 8, [32, 2], q80), ("q6k", 14, [256, 1], q6k),
        ("experts", 2, [32, 2, 3], q40x3),
        ("q4k", 12, [256, 1], zeros(UInt8, 144))]
    path = write_gguf(joinpath(dir, "all.gguf"), meta, tensors; alignment = 64)
    f = G.GGUFFile(path)

    @testset "header and metadata" begin
        @test f.version == 3
        for (k, v) in meta
            @test G.metadata(f, k) == v
            @test typeof(G.metadata(f, k)) == typeof(v)
        end
        @test G.metadata(f, "t.missing", :fallback) === :fallback
        @test length(f.tensors) == length(tensors)
    end

    @testset "unquantized tensors" begin
        @test G.tensor(f, "dense") == dense
        @test G.tensor(f, "half") == Float32[1, 2.5, -3, 0]
        @test G.tensor(f, "half"; T = Float64) isa Vector{Float64}
        @test G.tensor(f, "brain") == bf16vals
    end

    @testset "quantized tensors are zero-copy views" begin
        for (name, QT, bytes, nrow) in (("q40", Wink.Quant.Q4_0Matrix, q40, 32),
                ("q41", Wink.Quant.Q4_1Matrix, q41, 32),
                ("q80", Wink.Quant.Q8_0Matrix, q80, 32),
                ("q6k", Wink.Quant.Q6_KMatrix, q6k, 256))
            A = G.tensor(f, name)
            @test A isa QT
            @test A.data isa SubArray && parent(A.data) === f.data
            @test Array(A) == Array(QT(bytes, nrow, size(A, 2)))
        end
        bytes, dims, typ = G.raw_tensor(f, "experts")
        @test dims == [32, 2, 3] && typ == 2
        @test bytes == q40x3
        @test length(G.raw_tensor(f, "dense")[1]) == 12 * 4
        @test length(G.raw_tensor(f, "half")[1]) == 4 * 2
        @test G.raw_tensor(f, "q41")[1] == q41
    end

    @testset "failure modes" begin
        @test_throws ErrorException G.tensor(f, "nope")
        @test_throws ErrorException G.raw_tensor(f, "nope")
        @test_throws ErrorException G.tensor(f, "q4k")          # unsupported type
        @test_throws ErrorException G.raw_tensor(f, "q4k")
        @test_throws ErrorException G.raw_tensor(f, "brain")    # bf16: no raw path
        @test_throws ErrorException G.tensor(f, "experts")      # 3-D quantized

        bad = joinpath(dir, "bad.gguf")
        write(bad, UInt32(0x12345678), zeros(UInt8, 32))
        @test_throws ErrorException G.GGUFFile(bad)
        old = write_gguf(joinpath(dir, "v1.gguf"), meta, Any[]; version = 1)
        @test_throws ErrorException G.GGUFFile(old)
        v2 = write_gguf(joinpath(dir, "v2.gguf"), meta, Any[]; version = 2)
        @test G.GGUFFile(v2).version == 2
    end
end
