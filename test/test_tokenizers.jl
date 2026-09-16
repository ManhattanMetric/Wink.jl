# Both pure tokenizer families against llama.cpp's tokenizations of the same
# synthetic vocabularies (recorded goldens), plus the behaviors worth pinning
# down by name: special-token partitioning, byte fallback, rendering.

@testset "tokenizers" begin
    dir = mktempdir()

    @testset "$name matches llama.cpp" for (name, build, TK) in (
            ("spm", p -> tiny_gemma3!(p), Wink.SPMTokenizer),
            ("bpe", p -> tiny_olmoe!(p), Wink.BPETokenizer))
        g = GOLDEN_TOKENIZERS[name]
        path = build(joinpath(dir, "$name.gguf"))
        @test fnv1a(read(path)) == g.fingerprint      # fixture drift is loud
        t = TK.Tokenizer(Wink.GGUF.GGUFFile(path))
        for (text, add, parse, ids) in g.cases
            @test TK.tokenize(t, text; add_special = add, parse_special = parse) == ids
        end
        @test [TK.piece(t, id) for id in 0:(length(g.pieces) - 1)] == g.pieces
    end

    @testset "SentencePiece behaviors" begin
        t = Wink.SPMTokenizer.Tokenizer(
            Wink.GGUF.GGUFFile(tiny_gemma3!(joinpath(dir, "spm2.gguf"))))
        S = Wink.SPMTokenizer
        @test S.tokenize(t, "hello world"; add_special = true)[1] == t.bos
        @test S.tokenize(t, "") == Int[]
        # control tokens parse only under parse_special; user-defined space
        # runs partition the text either way
        @test t.eos in S.tokenize(t, "a<eos>b"; parse_special = true)
        @test !(t.eos in S.tokenize(t, "a<eos>b"; parse_special = false))
        spaces = t.id_of["  "]
        @test spaces in S.tokenize(t, "hello  world"; parse_special = false)
        # characters outside the vocabulary fall back to byte tokens
        ids = S.tokenize(t, "é")
        @test ids == [t.id_of["<0xC3>"], t.id_of["<0xA9>"]]
        @test S.detokenize(t, ids) == "é"
        # rendering: ▁ becomes space; control pieces only when asked
        @test S.detokenize(t, S.tokenize(t, "the function julia")) ==
              "the function julia"
        @test S.piece(t, t.eos; special = false) == ""
        @test S.piece(t, t.eos) == "<eos>"
    end

    @testset "BPE behaviors" begin
        f = Wink.GGUF.GGUFFile(tiny_olmoe!(joinpath(dir, "bpe2.gguf")))
        t = Wink.BPETokenizer.Tokenizer(f)
        B = Wink.BPETokenizer
        for text in ("hello world", "it's julia's world", "café ☃", "a\nb")
            @test B.detokenize(t, B.tokenize(t, text)) == text
        end
        eot = t.id_of["<|endoftext|>"]
        @test B.tokenize(t, "<|endoftext|>hi"; parse_special = true)[1] == eot
        @test !(eot in B.tokenize(t, "<|endoftext|>hi"; parse_special = false))
        @test B.piece(t, eot; special = false) == ""
        @test B.piece(t, t.id_of["   "]) == "   "    # user-defined, stored raw

        # an unrecognized pre-tokenizer warns and falls back to GPT-2's
        meta, _ = InferenceFixtures.bpe_vocab_meta()
        meta = [k == "tokenizer.ggml.pre" ? (k => "mystery") : (k => v) for (k, v) in meta]
        odd = Wink.GGUF.GGUFFile(write_gguf(joinpath(dir, "odd.gguf"), meta, Any[]))
        t2 = @test_logs (:warn, r"unrecognized BPE pre-tokenizer") B.Tokenizer(odd)
        @test B.tokenize(t2, "hello world") == B.tokenize(t, "hello world")
    end
end
