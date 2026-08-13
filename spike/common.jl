# Shared helpers for the spike scripts (chat.jl onward). The two original
# validation scripts (inprocess.jl, inprocess_gen.jl) intentionally stay
# self-contained.

include(joinpath(@__DIR__, "LibLlama.jl"))
import .LibLlama as L

# Overwrite one field of an isbits struct held in a Ref (the C-style
# "defaults, then tweak" idiom without a 37-argument constructor).
function poke!(r::Ref{T}, name::Symbol, val) where {T}
    i = Base.fieldindex(T, name)
    FT = fieldtype(T, i)
    GC.@preserve r begin
        p = Base.unsafe_convert(Ptr{T}, r)
        unsafe_store!(Ptr{FT}(Ptr{UInt8}(p) + fieldoffset(T, i)), convert(FT, val))
    end
    return r
end

function load_model(path::AbstractString; n_gpu_layers::Integer = 99)
    isfile(path) || error("model not found at $path")
    mp = Ref(L.llama_model_default_params())
    poke!(mp, :n_gpu_layers, Int32(n_gpu_layers))
    model = L.llama_model_load_from_file(path, mp[])
    model == C_NULL && error("failed to load $path")
    return model
end

function new_context(model; n_ctx::Integer = 4096, n_batch::Integer = 512)
    cp = Ref(L.llama_context_default_params())
    poke!(cp, :n_ctx, UInt32(n_ctx))
    poke!(cp, :n_batch, UInt32(n_batch))
    ctx = L.llama_init_from_model(model, cp[])
    ctx == C_NULL && error("failed to create context")
    return ctx
end

function tokenize(vocab, text::AbstractString; add_special::Bool, parse_special::Bool = true)
    buf = Vector{Int32}(undef, ncodeunits(text) + 32)
    n = L.llama_tokenize(vocab, text, ncodeunits(text), buf, length(buf),
        add_special, parse_special)
    if n < 0
        resize!(buf, -n)
        n = L.llama_tokenize(vocab, text, ncodeunits(text), buf, length(buf),
            add_special, parse_special)
    end
    n < 0 && error("tokenize failed")
    return resize!(buf, n)
end

function piece(vocab, tok::Integer)
    buf = Vector{UInt8}(undef, 256)
    len = L.llama_token_to_piece(vocab, tok, buf, length(buf), 0, true)
    return String(buf[1:max(len, 0)])
end

decode!(ctx, tokens::Vector{Int32}) = GC.@preserve tokens begin
    L.llama_decode(ctx, L.llama_batch_get_one(pointer(tokens), length(tokens))) == 0 ||
        error("llama_decode failed")
end
