# Live constrained tool-calling: the model cannot emit a malformed call.
#
# Two modes over the same grammar (compiled from the tool schemas by
# grammar.jl, targeting gemma-4's canonical call syntax):
#   forced — llama_sampler_init_grammar from position 0: the model MUST
#            produce one well-formed call. For when Wink knows a tool call
#            is required.
#   lazy   — llama_sampler_init_grammar_lazy with the "<|tool_call>" trigger:
#            free text until the model chooses to open a call, then clamped
#            to validity. The production shape for the backend.
#
# Run: julia --project=spike spike/toolcall.jl [path-to-gguf]

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "render.jl"))
include(joinpath(@__DIR__, "grammar.jl"))

const MODEL = get(ARGS, 1, expanduser(
    "~/.lmstudio/models/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf"))

D(pairs...) = Dict{String, Any}(pairs...)
const TOOLS = [
    D("function" => D("name" => "search_docs",
        "description" => "Semantic search over the Julia session's documentation.",
        "parameters" => D("type" => "object",
            "properties" => D(
                "query" => D("type" => "string",
                    "description" => "What to look for."),
                "limit" => D("type" => "integer",
                    "description" => "Maximum number of results.")),
            "required" => ["query", "limit"]))),
    D("function" => D("name" => "get_doc",
        "description" => "Fetch the docstring for a named binding.",
        "parameters" => D("type" => "object",
            "properties" => D("name" => D("type" => "string",
                "description" => "Function, type, or module name.")),
            "required" => ["name"]))),
]
const GRAMMAR = tool_call_grammar(TOOLS)

L.llama_backend_init()
print("loading $(basename(MODEL)) … ")
model = load_model(MODEL)
vocab = L.llama_model_get_vocab(model)
println("ok")

function build_chain(vocab; grammar = nothing, lazy = false)
    chain = L.llama_sampler_chain_init(L.llama_sampler_chain_default_params())
    if grammar !== nothing
        g = if lazy
            trigger = "<|tool_call>"
            triggers = [trigger]
            ptrs = [Ptr{Cchar}(pointer(t)) for t in triggers]
            GC.@preserve triggers ptrs begin
                L.llama_sampler_init_grammar_lazy(vocab, grammar, "root",
                    ptrs, length(ptrs), C_NULL, 0)
            end
        else
            L.llama_sampler_init_grammar(vocab, grammar, "root")
        end
        g == C_NULL && error("grammar sampler rejected the grammar")
        L.llama_sampler_chain_add(chain, g)   # mask first, then sample
    end
    L.llama_sampler_chain_add(chain, L.llama_sampler_init_top_k(Int32(40)))
    L.llama_sampler_chain_add(chain, L.llama_sampler_init_top_p(0.95f0, 1))
    L.llama_sampler_chain_add(chain, L.llama_sampler_init_temp(0.7f0))
    L.llama_sampler_chain_add(chain, L.llama_sampler_init_dist(UInt32(42)))
    return chain
end

function run_case(label, user_text; grammar = nothing, lazy = false)
    ctx = new_context(model; n_ctx = 8192)
    chain = build_chain(vocab; grammar, lazy)
    msgs = [(role = "system",
             content = "You are Wink, an assistant inside a Julia REPL. " *
                       "Use the provided tools when they help."),
            (role = "user", content = user_text)]
    prompt = render_chat(Gemma4Template(), msgs; tools = TOOLS, add_assistant = true)
    toks = tokenize(vocab, prompt; add_special = true)
    decode!(ctx, toks)
    out = IOBuffer()
    next = Int32[0]
    t0 = time()
    n = 0
    while n < 300
        tok = L.llama_sampler_sample(chain, ctx, Int32(-1))
        next[1] = tok
        L.llama_vocab_is_eog(vocab, tok) && break
        print(out, piece(vocab, tok))
        n += 1
        decode!(ctx, next)
    end
    text = String(take!(out))
    println("\n=== $label ===")
    println("user: ", user_text)
    println("raw:  ", repr(text))
    call = parse_tool_call(text)
    call === nothing ? println("call: (none — plain text answer)") :
    println("call: ", call.name, "  args: ", call.args)
    println("$(n) tokens, $(round(n / (time() - t0); digits = 1)) tok/s")
    L.llama_sampler_free(chain)
    L.llama_free(ctx)
    return call
end

# forced: a well-formed call is the only representable output
c = run_case("FORCED grammar", "Look up how broadcasting works in Julia.";
    grammar = GRAMMAR)
@assert c !== nothing && c.name in ("search_docs", "get_doc")
@assert haskey(c.args, c.name == "search_docs" ? "query" : "name")

# lazy: the model chooses — tool-shaped question should call
c = run_case("LAZY grammar, tool-shaped question",
    "Find documentation about how broadcasting works."; grammar = GRAMMAR, lazy = true)

# lazy: small talk should stay plain text
c = run_case("LAZY grammar, small talk",
    "Just say hello — no tools needed."; grammar = GRAMMAR, lazy = true)

L.llama_model_free(model)
L.llama_backend_free()
println("\nclean shutdown ok")
