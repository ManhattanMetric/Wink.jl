@testset "config" begin
    cfg = Wink.WinkConfig()

    withenv("ANTHROPIC_API_KEY" => "sk-test", "OPENAI_API_KEY" => nothing) do
        PT.load_api_keys!()
        Wink.detect_providers!(cfg)
    end
    @test cfg.chat_model == "claude-opus-5"
    @test cfg.embed_model == "nomic-embed-text"

    withenv("ANTHROPIC_API_KEY" => nothing, "OPENAI_API_KEY" => "sk-test") do
        PT.load_api_keys!()
        Wink.detect_providers!(cfg)
    end
    @test cfg.chat_model == PT.MODEL_CHAT
    @test cfg.embed_model == "text-embedding-3-small"

    withenv("ANTHROPIC_API_KEY" => nothing, "OPENAI_API_KEY" => nothing) do
        PT.load_api_keys!()
        Wink.detect_providers!(cfg)
    end
    @test cfg.chat_model == "llama3.1"
    @test haskey(PT.MODEL_REGISTRY, "llama3.1")
    @test cfg.embed_model == "nomic-embed-text"

    withenv("ANTHROPIC_API_KEY" => "a", "OPENAI_API_KEY" => "b") do
        PT.load_api_keys!()
        Wink.detect_providers!(cfg)
    end
    @test cfg.chat_model == "claude-opus-5"
    @test cfg.embed_model == "text-embedding-3-small"

    # restore the ambient environment's keys for later suites
    PT.load_api_keys!()

    # configure! registers unknown models when a schema is supplied...
    Wink.configure!(chat_model = "wink-test-model-xyz", chat_schema = PT.OllamaSchema())
    @test Wink.CONFIG.chat_model == "wink-test-model-xyz"
    @test haskey(PT.MODEL_REGISTRY, "wink-test-model-xyz")
    # ...and warns when it cannot fix routing
    @test_logs (:warn,) match_mode = :any Wink.configure!(chat_model = "wink-unknown-abc")

    # configure! with chat_api_base auto-registers the model for an
    # OpenAI-compatible local server (no warning, CustomOpenAISchema routing)
    @test_logs Wink.configure!(chat_model = "wink-local-model-xyz",
        chat_api_base = "http://localhost:1234/v1")
    @test Wink.CONFIG.chat_api_base == "http://localhost:1234/v1"
    @test Wink.CONFIG.chat_model == "wink-local-model-xyz"
    @test PT.MODEL_REGISTRY["wink-local-model-xyz"].schema isa PT.CustomOpenAISchema
    @test Wink._model_schema("wink-local-model-xyz") isa PT.CustomOpenAISchema
    Wink.configure!(chat_api_base = "")
    @test isempty(Wink.CONFIG.chat_api_base)

    # broken-dual-stack diagnostics: the hint stays quiet for anything but
    # the connect-timeout shape (the positive case needs a live resolver and
    # is exercised manually)
    @test Wink._connect_hint(ErrorException("boom")) === nothing
    @test Wink._connect_hint(Wink.HTTP.Exceptions.ConnectError(
        "https://api.anthropic.com/v1/messages",
        ErrorException("connection refused"))) === nothing
    # the IPv4-first override installs cleanly and is idempotent
    redirect_stderr(devnull) do
        @test Wink.prefer_ipv4!() === nothing
        @test Wink.prefer_ipv4!() === nothing
    end

    # claude models newer than the pinned PT registry must still route to
    # Anthropic — a registry miss would silently fall through to PT's
    # default OpenAI schema ("api_key cannot be empty" with a good
    # ANTHROPIC_API_KEY set)
    @test Wink._model_schema("claude-opus-5") isa PT.AnthropicSchema
    @test PT.MODEL_REGISTRY["claude-opus-5"].schema isa PT.AnthropicSchema
    @test Wink._model_schema("claude-fable-5-unheard-of") isa PT.AnthropicSchema
    # known registry entries keep their spec schema
    @test Wink._model_schema("claude-3-5-sonnet-latest") isa PT.AbstractAnthropicSchema

    # ...and the same for embedding models via embed_api_base
    @test_logs Wink.configure!(embed_model = "wink-local-embed-xyz",
        embed_api_base = "http://localhost:1234/v1")
    @test Wink.CONFIG.embed_api_base == "http://localhost:1234/v1"
    @test PT.MODEL_REGISTRY["wink-local-embed-xyz"].schema isa PT.CustomOpenAISchema
    Wink.configure!(embed_api_base = "")
    @test isempty(Wink.CONFIG.embed_api_base)

    Wink.autoeval!(true)
    @test Wink.CONFIG.autoeval
    Wink.autoeval!(false)
    @test !Wink.CONFIG.autoeval

    Wink.configure!(debug = true)
    @test Wink.CONFIG.debug
    Wink.configure!(debug = false)
    @test !Wink.CONFIG.debug

    # put global config back in a sane state for later suites
    Wink.detect_providers!(Wink.CONFIG)
end
