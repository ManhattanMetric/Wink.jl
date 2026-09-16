# The in-process forward passes on tiny synthetic models: logits against
# llama.cpp's on the same files (recorded goldens), and the internal
# invariants — a prompt processed at once equals it processed in chunks or
# token by token (KV cache, sliding windows), quantized storage tracks its
# exact dense twin, device movement preserves results.

const ENGINES = Dict("gemma3" => Wink.Gemma3, "gemma4" => Wink.Gemma4,
    "olmoe" => Wink.OLMoE)

cos_sim(a, b) = sum(a .* b) / sqrt(sum(abs2, a) * sum(abs2, b))

function build_model(dir, name)
    path = joinpath(dir, name * ".gguf")
    if name == "gemma3"
        tiny_gemma3!(path)
    elseif name == "gemma3_q"
        tiny_gemma3!(path; quant = true)
    elseif name == "gemma4"
        tiny_gemma4!(path)
    elseif name == "olmoe"
        tiny_olmoe!(path)
    else
        tiny_olmoe!(path; quant = true)
    end
    return path
end

engine_for(name) = ENGINES[first(split(name, "_"))]
logits_of(eng, m, toks) = Matrix{Float32}(eng.forward(m, toks))

@testset "model forward passes" begin
    dir = mktempdir()
    loaded = Dict{String, Any}()
    for name in keys(GOLDEN_MODELS)
        path = build_model(dir, name)
        loaded[name] = (path, engine_for(name).load_model(Wink.GGUF.GGUFFile(path)))
    end

    @testset "$name matches llama.cpp" for (name, g) in GOLDEN_MODELS
        path, m = loaded[name]
        eng = engine_for(name)
        @test fnv1a(read(path)) == g.fingerprint
        L = logits_of(eng, m, g.tokens)
        dense = !(endswith(name, "_q") || name == "gemma4")
        tol = dense ? 0.99999 : 0.999
        @test cos_sim(L[:, 6], g.logits_mid) > tol
        @test cos_sim(L[:, end], g.logits_last) > tol
        top1 = count(i -> argmax(L[:, i]) - 1 == g.argmax[i], axes(L, 2))
        # quantized kernels carry requantization noise; allow one near-tie
        @test top1 >= length(g.tokens) - (dense ? 0 : 1)
    end

    @testset "$name: cached and chunked decoding agree" for name in
            ("gemma3", "gemma4", "olmoe_q")
        _, m = loaded[name]
        eng = engine_for(name)
        toks = GOLDEN_MODELS[name].tokens
        full = logits_of(eng, m, toks)
        # token by token past the 4-token sliding window
        c = eng.KVCache(m; capacity = length(toks))
        stepped = reduce(hcat, [vec(Matrix{Float32}(eng.step!(m, c, [t]))) for t in toks])
        @test c.n == length(toks)
        # single-token and batched decoding take different code paths (fused
        # kernels vs grouped matmuls), so rounding differs slightly; with a
        # MoE, an expert near-tie can then flip a routing choice at one
        # position — allow exactly that, and nothing looser
        agree(M) = count(i -> maximum(abs, M[:, i] .- full[:, i]) < 1.0e-3, axes(full, 2))
        @test agree(stepped) >= length(toks) - 1
        @test all(i -> cos_sim(stepped[:, i], full[:, i]) > 0.9999, axes(full, 2))
        @test all(i -> argmax(stepped[:, i]) == argmax(full[:, i]), axes(full, 2))
        # uneven chunks
        c = eng.KVCache(m; capacity = length(toks))
        a = Matrix{Float32}(eng.step!(m, c, toks[1:5]))
        b = Matrix{Float32}(eng.step!(m, c, toks[6:end]))
        @test agree(hcat(a, b)) >= length(toks) - 1
        @test_throws ErrorException eng.step!(m, c, [toks[1]])     # cache full
        # greedy generation: first token is the prompt's argmax
        gen = eng.generate(m, toks; max_tokens = 3)
        @test length(gen) == 3 && gen[1] == argmax(full[:, end]) - 1
        @test isempty(eng.generate(m, toks; max_tokens = 3, eog = [gen[1]]))
    end

    @testset "quantized storage tracks the dense twin" begin
        for (dense, quant) in (("gemma3", "gemma3_q"), ("olmoe", "olmoe_q"))
            eng = engine_for(dense)
            toks = GOLDEN_MODELS[dense].tokens
            A = logits_of(eng, loaded[dense][2], toks)
            B = logits_of(eng, loaded[quant][2], toks)
            @test minimum(cos_sim(A[:, i], B[:, i]) for i in axes(A, 2)) > 0.98
        end
        @test loaded["olmoe_q"][2].layers[1].down_exps isa Vector   # q4_1 slabs
        @test loaded["olmoe_q"][2].layers[2].down_exps isa Wink.Quant.Q4_0Stack
        @test loaded["olmoe"][2].layers[1].down_exps isa Wink.OLMoE.DenseStack
    end

    @testset "gemma-4 specifics" begin
        _, m = loaded["gemma4"]
        L = logits_of(Wink.Gemma4, m, GOLDEN_MODELS["gemma4"].tokens)
        @test maximum(abs, L) <= 30                                 # softcap
        @test count(L -> !L.is_swa, m.layers) == 1
        @test m.layers[end].wv === nothing                          # K doubles as V
    end

    # The GPU forward path's fused routing and MoE kernels, run on the
    # KernelAbstractions CPU backend against the host computation they replace
    @testset "gemma-4 device kernels (CPU backend)" begin
        _, m = loaded["gemma4"]
        G4 = Wink.Gemma4
        cpu = Wink.Quant.KernelAbstractions.CPU()
        L = m.layers[1]
        r = InferenceFixtures.SplitMix(0x6b)
        T, K, ne = 3, m.n_expert_used, m.n_embd
        nex, nf = size(L.gate_inp, 2), L.down_exps.nrow
        X = Float32[InferenceFixtures.sym(r, 1.0f0) for _ in 1:ne, _ in 1:T]

        p = zeros(Float32, nex, T)
        G4._dense_matvec!(cpu)(p, L.gate_inp, X; ndrange = (nex, T))
        @test p ≈ L.gate_inp' * X rtol = 1.0e-5
        p = exp.(p .- maximum(p; dims = 1))
        p ./= sum(p; dims = 1)

        sel = zeros(Int32, K, T)
        w = zeros(Float32, K, T)
        G4._router_topk!(cpu)(sel, w, p, L.down_exps_s, Int32(K), 6.103515625f-5;
            ndrange = T)
        for t in 1:T
            top = partialsortperm(p[:, t], 1:K; rev = true)
            @test sort(sel[:, t]) == sort(top)
            wsum = max(sum(p[top, t]), 6.103515625f-5)
            @test w[:, t] ≈ [p[e, t] / wsum * L.down_exps_s[e] for e in sel[:, t]] rtol = 1.0e-5
        end

        gu, dn = L.gate_up_exps, L.down_exps
        act = zeros(Float32, nf, K * T)
        moe = zeros(Float32, ne, T)
        G4._moe_act!(cpu)(act, gu.data, sel, X, Int32(gu.nrow ÷ 32), Int32(nf),
            Int32(K), Wink.Quant.perexp(gu); ndrange = (nf, K * T))
        G4._moe_down!(cpu)(moe, dn.data, sel, w, act, Int32(dn.nrow ÷ 32),
            Int32(K), Wink.Quant.perexp(dn); ndrange = (ne, T))
        ref = zeros(Float32, ne, T)
        for t in 1:T, k in 1:K
            e = sel[k, t]
            GU = Array(gu[e])' * X[:, t]
            a = G4.gelu_tanh.(GU[1:nf]) .* GU[(nf + 1):(2nf)]
            ref[:, t] .+= (Array(dn[e])' * a) .* w[k, t]
        end
        @test moe ≈ ref rtol = 1.0e-4
    end

    @testset "device movement round-trips" begin
        for name in ("gemma3", "gemma4", "olmoe")
            _, m = loaded[name]
            eng = engine_for(name)
            toks = GOLDEN_MODELS[name].tokens
            moved = Wink.Quant.Adapt.adapt(Array, m)
            @test logits_of(eng, moved, toks) ≈ logits_of(eng, m, toks) rtol = 1.0e-4
        end
    end

    @testset "architecture guards" begin
        f3 = Wink.GGUF.GGUFFile(loaded["gemma3"][1])
        @test_throws ErrorException Wink.Gemma4.load_model(f3)
        @test_throws ErrorException Wink.OLMoE.load_model(f3)
        @test_throws ErrorException Wink.Gemma3.load_model(
            Wink.GGUF.GGUFFile(loaded["olmoe"][1]))
    end
end
