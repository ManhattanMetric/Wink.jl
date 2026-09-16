# Quantized weight arrays: dequantization against ggml's own (recorded
# goldens), every mul! path against dense BLAS on the exact dequantization,
# the KernelAbstractions device kernels run on the CPU backend, and the
# supporting machinery (activation quantization, expert stacks, the worker
# pool, Adapt).

const Q = Wink.Quant
const KA = Wink.Quant.KernelAbstractions

# deterministic activations
activations(r, n, T) = Float32[InferenceFixtures.sym(r, 1.0f0) for _ in 1:n, _ in 1:T]

cosine(a, b) = sum(a .* b) / sqrt(sum(abs2, a) * sum(abs2, b))

@testset "quantized arrays" begin
    @testset "dequantization matches ggml" begin
        for (kind, QT) in ((:q4_0, Q.Q4_0Matrix), (:q4_1, Q.Q4_1Matrix),
                (:q8_0, Q.Q8_0Matrix), (:q6_K, Q.Q6_KMatrix))
            typ, qk, _ = QBLOCK[kind]
            bytes = qblocks(InferenceFixtures.SplitMix(0x0de9 + typ), kind, qk, 1)
            @test vec(Array(QT(bytes, qk, 1))) == GOLDEN_DEQUANT[kind]
        end
    end

    @testset "$kind" for (kind, QT, nrow, ncol, kernel) in (
            (:q4_0, Q.Q4_0Matrix, 64, 96, Q._q4_mul_kernel!),
            (:q4_1, Q.Q4_1Matrix, 64, 96, nothing),
            (:q8_0, Q.Q8_0Matrix, 64, 96, Q._q8_mul_kernel!),
            (:q6_K, Q.Q6_KMatrix, 256, 80, Q._q6_mul_kernel!))
        r = InferenceFixtures.SplitMix(0x5eed)
        bytes = qblocks(r, kind, nrow, ncol)
        A = QT(bytes, nrow, ncol)
        D = Array(A)                       # exact dequantization via getindex
        @test size(A) == (nrow, ncol)
        @test D isa Matrix{Float32} && all(isfinite, D)
        @test A[nrow, ncol] == D[end, end]

        X = activations(r, nrow, 3)
        ref = D' * X
        # the SDOT kernels requantize activations to int8 (llama.cpp's own
        # numerics); every other path computes the exact dequantized product
        exact = !(Q.USE_SDOT && kind in (:q4_0, :q6_K))
        for x in (X, X[:, 1:1])
            got = A' * x
            want = D' * x
            if exact
                @test got ≈ want rtol = 1.0e-4
            else
                @test cosine(vec(got), vec(want)) > 0.999
                @test maximum(abs, got .- want) / maximum(abs, want) < 0.05
            end
        end

        if kernel !== nothing              # the GPU code path, on the CPU
            Yk = similar(ref)
            Q._mul_ka!(kernel, KA.CPU(), Yk, A, X, nrow ÷ QBLOCK[kind][2])
            @test Yk ≈ ref rtol = 1.0e-4
        end

        @test_throws DimensionMismatch A' * zeros(Float32, nrow + QBLOCK[kind][2], 1)
        @test_throws ErrorException QT(bytes[1:(end - 1)], nrow, ncol)
        @test_throws ErrorException QT(bytes, nrow + 1, ncol)

        @test similar(A, Float32, (2, 3)) isa Matrix{Float32}
        @test KA.get_backend(A) isa KA.CPU
        # zero-copy views (as read from an mmap) and device movement
        V = QT(view(bytes, 1:length(bytes)), nrow, ncol)
        @test KA.get_backend(V) isa KA.CPU
        moved = Q.Adapt.adapt(Array, V)
        @test moved isa QT && moved.data isa Vector{UInt8}
        @test Array(moved) == D
    end

    @testset "activation quantization" begin
        X = zeros(Float32, 64, 2)
        X[1:32, 2] .= range(-1.0f0, 1.0f0; length = 32)
        a = Q.quantize_q8!(Q.Q8Act(64, 2), X)
        @test a.d[:, 1] == zeros(Float32, 2)          # all-zero blocks stay zero
        @test all(==(0), a.q[:, 1])
        @test a.d[1, 2] ≈ 1.0f0 / 127
        @test a.q[1, 2] == -127 && a.q[32, 2] == 127
        @test Q._q8_scratch(64, 2) === Q._q8_scratch(64, 2)   # reused per task
    end

    @testset "q4_dot" begin
        r = InferenceFixtures.SplitMix(0xd07)
        bytes = qblocks(r, :q4_0, 64, 3)
        D = Array(Q.Q4_0Matrix(bytes, 64, 3))
        X = activations(r, 64, 2)
        @test Q.q4_dot(bytes, 2 * 2 * 18, X, 2, 2) ≈ sum(D[:, 3] .* X[:, 2]) rtol = 1.0e-5
    end

    @testset "expert stacks" begin
        r = InferenceFixtures.SplitMix(0x57ac)
        slabs = [qblocks(r, :q4_0, 32, 16) for _ in 1:3]
        S = Q.Q4_0Stack(reduce(vcat, slabs), 32, 16, 3)
        @test length(S) == 3
        @test Q.perexp(S) == length(slabs[1])
        @test Array(S[2]) == Array(Q.Q4_0Matrix(slabs[2], 32, 16))
        @test S[3].data isa SubArray                  # zero-copy per-expert view
        moved = Q.Adapt.adapt(Array, S)
        @test moved.data isa Vector{UInt8} && Array(moved[1]) == Array(S[1])
    end

    @testset "worker pool" begin
        for n in (5, 1000)                            # serial and pooled sizes
            hits = zeros(Int, n)
            Q.parallel_cols(n) do lo, hi
                hits[lo:hi] .+= 1
            end
            @test all(==(1), hits)
        end
    end
end
