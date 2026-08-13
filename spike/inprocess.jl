# Feasibility spike: drive llama.cpp IN-PROCESS from Julia — no server, no
# subprocess. The model loads into this Julia process's address space and
# token generation is a plain Julia loop; sampling happens on the logits as an
# ordinary Julia array (greedy = argmax), which is the whole point: once the
# logits are visible from the session, everything from grammar-constrained
# tool calls to custom samplers is session-level Julia code.
#
# Bindings: hand-written ccall declarations matched to vendor/llama.h at tag
# b10405 (the vendored release). Struct layouts are transcribed field-for-
# field from that header — regenerate with Clang.jl against the header when
# this graduates from spike to backend.
#
# Run: julia --project=spike spike/inprocess.jl [path-to-gguf]

using Libdl

const VENDOR = joinpath(@__DIR__, "vendor", "llama-b10405")
const LIB = joinpath(VENDOR, "libllama.dylib")
const MODEL = get(ARGS, 1, expanduser(
    "~/.lmstudio/models/lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF/" *
    "NVIDIA-Nemotron-3-Nano-4B-Q8_0.gguf"))

isfile(LIB) || error("vendored libllama not found at $LIB")
isfile(MODEL) || error("model not found at $MODEL")
dlopen(LIB)   # @loader_path rpath resolves the ggml deps beside it

# ---- struct mirrors of vendor/llama.h (b10405) --------------------------------

struct LlamaModelParams
    devices::Ptr{Cvoid}
    tensor_buft_overrides::Ptr{Cvoid}
    n_gpu_layers::Int32
    split_mode::Int32
    load_mode::Int32
    main_gpu::Int32
    tensor_split::Ptr{Cfloat}
    progress_callback::Ptr{Cvoid}
    progress_callback_user_data::Ptr{Cvoid}
    kv_overrides::Ptr{Cvoid}
    vocab_only::Bool
    check_tensors::Bool
    use_extra_bufts::Bool
    no_host::Bool
    no_alloc::Bool
    load_mtp::Bool
end

struct LlamaContextParams
    n_ctx::UInt32
    n_batch::UInt32
    n_ubatch::UInt32
    n_seq_max::UInt32
    n_rs_seq::UInt32
    n_outputs_max::UInt32
    n_outputs_max_per_seq::UInt32
    n_threads::Int32
    n_threads_batch::Int32
    ctx_type::Int32
    rope_scaling_type::Int32
    pooling_type::Int32
    attention_type::Int32
    flash_attn_type::Int32
    rope_freq_base::Cfloat
    rope_freq_scale::Cfloat
    yarn_ext_factor::Cfloat
    yarn_attn_factor::Cfloat
    yarn_beta_fast::Cfloat
    yarn_beta_slow::Cfloat
    yarn_orig_ctx::UInt32
    defrag_thold::Cfloat
    cb_eval::Ptr{Cvoid}
    cb_eval_user_data::Ptr{Cvoid}
    type_k::Int32
    type_v::Int32
    abort_callback::Ptr{Cvoid}
    abort_callback_data::Ptr{Cvoid}
    embeddings::Bool
    offload_kqv::Bool
    no_perf::Bool
    op_offload::Bool
    swa_full::Bool
    kv_unified::Bool
    samplers::Ptr{Cvoid}
    n_samplers::Csize_t
    ctx_other::Ptr{Cvoid}
end

struct LlamaBatch
    n_tokens::Int32
    token::Ptr{Int32}
    embd::Ptr{Cfloat}
    pos::Ptr{Int32}
    n_seq_id::Ptr{Int32}
    seq_id::Ptr{Ptr{Int32}}
    logits::Ptr{Int8}
end

# Overwrite one field of an isbits struct held in a Ref (the C-style
# "defaults, then tweak" idiom without writing a 37-argument constructor).
function poke!(r::Ref{T}, name::Symbol, val) where {T}
    i = Base.fieldindex(T, name)
    FT = fieldtype(T, i)
    GC.@preserve r begin
        p = Base.unsafe_convert(Ptr{T}, r)
        unsafe_store!(Ptr{FT}(Ptr{UInt8}(p) + fieldoffset(T, i)), convert(FT, val))
    end
    return r
end

# ---- load ---------------------------------------------------------------------

ccall((:llama_backend_init, LIB), Cvoid, ())

mp = Ref(ccall((:llama_model_default_params, LIB), LlamaModelParams, ()))
poke!(mp, :n_gpu_layers, Int32(99))          # Metal: offload everything

print("loading $(basename(MODEL)) … ")
t0 = time()
model = ccall((:llama_model_load_from_file, LIB), Ptr{Cvoid},
    (Cstring, LlamaModelParams), MODEL, mp[])
model == C_NULL && error("llama_model_load_from_file failed")
println("done ($(round(time() - t0; digits = 1))s)")

vocab = ccall((:llama_model_get_vocab, LIB), Ptr{Cvoid}, (Ptr{Cvoid},), model)
n_vocab = ccall((:llama_vocab_n_tokens, LIB), Int32, (Ptr{Cvoid},), vocab)

cp = Ref(ccall((:llama_context_default_params, LIB), LlamaContextParams, ()))
poke!(cp, :n_ctx, UInt32(4096))
poke!(cp, :n_batch, UInt32(512))
ctx = ccall((:llama_init_from_model, LIB), Ptr{Cvoid},
    (Ptr{Cvoid}, LlamaContextParams), model, cp[])
ctx == C_NULL && error("llama_init_from_model failed")

# Layout sanity check: if our struct mirror is wrong, this readback won't be
# the 4096 we poked in.
n_ctx = ccall((:llama_n_ctx, LIB), UInt32, (Ptr{Cvoid},), ctx)
n_ctx == 4096 || error("context params layout mismatch: n_ctx read back as $n_ctx")
println("context ok (n_ctx = $n_ctx, n_vocab = $n_vocab)")

# ---- generate -----------------------------------------------------------------

piece(tok) = begin
    buf = Vector{UInt8}(undef, 256)
    len = ccall((:llama_token_to_piece, LIB), Int32,
        (Ptr{Cvoid}, Int32, Ptr{UInt8}, Int32, Int32, Bool),
        vocab, tok, buf, 256, 0, true)
    String(buf[1:max(len, 0)])
end

decode!(tokens::Vector{Int32}) = GC.@preserve tokens begin
    batch = ccall((:llama_batch_get_one, LIB), LlamaBatch,
        (Ptr{Int32}, Int32), tokens, length(tokens))
    ret = ccall((:llama_decode, LIB), Int32, (Ptr{Cvoid}, LlamaBatch), ctx, batch)
    ret == 0 || error("llama_decode failed with $ret")
end

prompt = "The capital of France is"
toks = Vector{Int32}(undef, 512)
n = ccall((:llama_tokenize, LIB), Int32,
    (Ptr{Cvoid}, Cstring, Int32, Ptr{Int32}, Int32, Bool, Bool),
    vocab, prompt, ncodeunits(prompt), toks, length(toks), true, true)
n > 0 || error("llama_tokenize failed with $n")
resize!(toks, n)
println("prompt: $(repr(prompt)) → $n tokens")

decode!(toks)                                # prompt eval

out = IOBuffer()
next = Int32[0]
n_gen = 0
t0 = time()
for _ in 1:32
    logits = ccall((:llama_get_logits_ith, LIB), Ptr{Float32},
        (Ptr{Cvoid}, Int32), ctx, Int32(-1))
    logits == C_NULL && error("no logits returned")
    v = unsafe_wrap(Array, logits, Int(n_vocab))
    tok = Int32(argmax(v) - 1)               # greedy sampling, in plain Julia
    ccall((:llama_vocab_is_eog, LIB), Bool, (Ptr{Cvoid}, Int32), vocab, tok) && break
    print(out, piece(tok))
    global n_gen += 1
    next[1] = tok
    decode!(next)
end
dt = time() - t0

println("completion: ", repr(String(take!(out))))
println("generated $n_gen tokens in $(round(dt; digits = 2))s → ",
    round(n_gen / dt; digits = 1), " tok/s (greedy, sampled in Julia)")

ccall((:llama_free, LIB), Cvoid, (Ptr{Cvoid},), ctx)
ccall((:llama_model_free, LIB), Cvoid, (Ptr{Cvoid},), model)
ccall((:llama_backend_free, LIB), Cvoid, ())
println("clean shutdown ok")
