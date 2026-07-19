@testset "rag" begin
    # corpus over the fixture only — no network
    chunks = Wink.build_corpus([TestPkg])
    @test !isempty(chunks)
    @test any(c -> c.binding == "TestPkg.greet", chunks)
    @test any(c -> c.binding == "TestPkg.ANSWER", chunks)
    # the deliberately long simulate docstring splits into suffixed chunks
    sim = filter(c -> startswith(c.binding, "TestPkg.simulate"), chunks)
    @test length(sim) >= 2
    @test any(c -> endswith(c.binding, "#1"), sim)
    @test all(c -> length(c.text) <= 2500, chunks)
    g = only(filter(c -> c.binding == "TestPkg.greet", chunks))
    @test occursin("friendly greeting", g.text)
    @test occursin("TestPkg.jl", g.file)
    @test g.line > 0

    # split_docstring behavior
    @test Wink.split_docstring("short") == ["short"]
    longtext = join(fill("para " * repeat("x", 300), 12), "\n\n")
    parts = Wink.split_docstring(longtext)
    @test length(parts) >= 3
    @test all(p -> length(p) <= 2500, parts)

    # cosine retrieval on a hand-built index (unit columns)
    unit(v) = Float32.(v ./ sqrt(sum(abs2, v)))
    E = hcat(unit([1.0, 0.0]), unit([0.9, 0.1]), unit([0.0, 1.0]))
    hand = [Wink.DocChunk("M", "M.alpha", "alpha()", "about alpha", "f.jl", 1),
        Wink.DocChunk("M", "M.almost", "almost()", "close to alpha", "f.jl", 2),
        Wink.DocChunk("M", "M.beta", "beta()", "about beta", "f.jl", 3)]
    idx = Wink.DocIndex(hand, E, "fake-model", Wink.modules_fingerprint())
    out = Wink.search_scores(idx, unit([1.0, 0.0]); k = 2)
    @test occursin("M.alpha", out)
    @test occursin("M.almost", out)
    @test !occursin("M.beta", out)
    first_hit = first(split(out, "\n"))
    @test occursin("M.alpha", first_hit)   # best match ranked first

    @test Wink.staleness_note(idx) == ""
    stale = Wink.DocIndex(hand, E, "fake-model", UInt64(0))
    @test occursin("reindex!", Wink.staleness_note(stale))

    # persistence round-trip through the scratch cache
    Wink.save_index(idx)
    f = Wink.index_cache_file("fake-model")
    @test isfile(f)
    old_embed = Wink.CONFIG.embed_model
    old_idx = Wink.DOC_INDEX[]
    try
        Wink.CONFIG.embed_model = "fake-model"
        Wink.DOC_INDEX[] = nothing
        loaded = Wink.load_index!(; io = devnull)
        @test loaded isa Wink.DocIndex
        @test length(loaded.chunks) == 3
        @test loaded.embeddings ≈ E
    finally
        Wink.CONFIG.embed_model = old_embed
        Wink.DOC_INDEX[] = old_idx
        rm(f; force = true)
    end

    # keyword fallback when no embedding provider is configured
    try
        Wink.CONFIG.embed_model = ""
        out = Wink.tool_search_docs("friendly greeting", "")
        @test occursin("semantic search unavailable", out)
        @test occursin("greet", out)
        @test startswith(Wink.tool_search_docs("   ", ""), "ERROR")
    finally
        Wink.CONFIG.embed_model = old_embed
    end

    # search_docs is registered as a tool
    tm = Wink.build_tool_map()
    @test haskey(tm, "search_docs")
    @test haskey(tm, "eval_code")
    @test haskey(tm, "edit_file")
end
