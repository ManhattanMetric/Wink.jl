# Semantic documentation search: docstring corpus -> embedding index -> cosine
# retrieval, with a keyword (apropos) fallback when no embedding provider
# exists. The index is built lazily on first use, persisted to a scratch cache,
# and refreshed explicitly via `reindex!` (never auto-embedded — cost surprise).

struct DocChunk
    mod::String
    binding::String
    sig::String
    text::String
    file::String
    line::Int
end

mutable struct DocIndex
    chunks::Vector{DocChunk}
    embeddings::Matrix{Float32}   # dim × n, unit-norm columns
    model::String
    modules_hash::UInt64
end

const DOC_INDEX = Ref{Union{Nothing, DocIndex}}(nothing)
const INDEX_FAILURE = Ref{Union{Nothing, String}}(nothing)

modules_fingerprint() =
    hash(sort!([string(nameof(m)) for m in values(Base.loaded_modules)]))

# ---- corpus ------------------------------------------------------------------

function docable_modules()
    seen = Base.IdSet{Module}()
    queue = Module[]
    for m in values(Base.loaded_modules)
        m in seen || (push!(seen, m); push!(queue, m))
    end
    out = Module[]
    while !isempty(queue)
        m = popfirst!(queue)
        push!(out, m)
        for n in names(m; all = true)
            startswith(string(n), "#") && continue
            isdefined(m, n) || continue
            v = try
                getglobal(m, n)
            catch
                continue
            end
            if v isa Module && !(v in seen) && v !== m && parentmodule(v) === m
                push!(seen, v)
                push!(queue, v)
            end
        end
    end
    return out
end

function split_docstring(text::AbstractString; maxlen::Int = 1800, target::Int = 1500)
    length(text) <= maxlen && return String[String(text)]
    parts = String[]
    buf = IOBuffer()
    for para in split(text, "\n\n")
        if position(buf) > 0 && position(buf) + length(para) > target
            push!(parts, String(take!(buf)))
        end
        position(buf) > 0 && write(buf, "\n\n")
        write(buf, para)
        while position(buf) > maxlen
            s = String(take!(buf))
            push!(parts, String(first(s, target)))
            write(buf, String(chop(s; head = target, tail = 0)))
        end
    end
    position(buf) > 0 && push!(parts, String(take!(buf)))
    return parts
end

"""
    build_corpus(mods = docable_modules()) -> Vector{DocChunk}

Collect every docstring of the given modules (via the docsystem's per-module
metadata) as retrieval chunks, splitting oversized docstrings at paragraph
boundaries.
"""
function build_corpus(mods::Vector{Module} = docable_modules())
    chunks = DocChunk[]
    for m in mods
        meta = try
            Base.Docs.meta(m; autoinit = false)
        catch
            nothing
        end
        meta === nothing && continue
        for (binding, multidoc) in meta
            binding isa Base.Docs.Binding || continue
            for sig in multidoc.order
                ds = get(multidoc.docs, sig, nothing)
                ds === nothing && continue
                text = try
                    strip(Markdown.plain(Base.Docs.parsedoc(ds)))
                catch
                    continue
                end
                isempty(text) && continue
                file = string(get(ds.data, :path, ""))
                line = Int(something(get(ds.data, :linenumber, nothing), 0))
                sigstr = sig === Union{} ? "" : string(sig)
                pieces = split_docstring(String(text))
                for (i, piece) in enumerate(pieces)
                    suffix = length(pieces) == 1 ? "" : "#$(i)"
                    push!(chunks,
                        DocChunk(string(m), string(binding) * suffix, sigstr,
                            piece, file, line))
                end
            end
        end
    end
    return chunks
end

# ---- embedding ---------------------------------------------------------------

chunk_embed_text(c::DocChunk) = string(c.binding, "\n", c.text)

# All embedding traffic funnels through here so custom OpenAI-compatible
# servers (CONFIG.embed_api_base) route the same way chat does.
function _aiembed_call(docs, model::AbstractString)
    api_kwargs, key_kwargs = custom_server_kwargs(
        _model_schema(String(model)), CONFIG.embed_api_base)
    return PT.aiembed(docs, LinearAlgebra.normalize; model = String(model),
        verbose = false, api_kwargs, key_kwargs...)
end

function embed_query(model::AbstractString, query::AbstractString)
    LOCAL_EMBED[] === nothing ||
        return vec(_local_embed_texts([String(first(query, 6000))]))
    msg = _aiembed_call(String(first(query, 6000)), model)
    return Float32.(msg.content)
end

function embed_texts(texts::Vector{String}; model::AbstractString = CONFIG.embed_model,
        io::IO = CONFIG.status_io)
    LOCAL_EMBED[] === nothing ||
        return _local_embed_texts(String[first(t, 6000) for t in texts]; io)
    isempty(model) && throw(WinkResolveError("no embedding model configured"))
    cols = Vector{Vector{Float32}}()
    batch = 64
    for (bi, lo) in enumerate(1:batch:length(texts))
        hi = min(lo + batch - 1, length(texts))
        docs = String[first(t, 6000) for t in texts[lo:hi]]
        msg = _aiembed_call(docs, model)
        M = msg.content
        M isa AbstractVector && (M = reshape(Float32.(M), :, 1))
        for j in axes(M, 2)
            push!(cols, Float32.(M[:, j]))
        end
        bi % 5 == 0 && status(io, "embedded $hi/$(length(texts)) docstrings")
    end
    return reduce(hcat, cols)
end

# ---- index lifecycle ---------------------------------------------------------

index_cache_file(model::AbstractString) =
    joinpath(Scratch.@get_scratch!("docindex"),
        "index-v1-$(VERSION.major).$(VERSION.minor)-" *
        replace(String(model), r"[^A-Za-z0-9._-]" => "_") * ".jls")

function save_index(idx::DocIndex)
    try
        Serialization.serialize(index_cache_file(idx.model), idx)
    catch e
        @debug "Wink: could not persist the doc index" exception = e
    end
    return nothing
end

function load_index!(; io::IO = CONFIG.status_io)
    f = index_cache_file(CONFIG.embed_model)
    isfile(f) || return nothing
    idx = try
        Serialization.deserialize(f)
    catch
        return nothing
    end
    (idx isa DocIndex && idx.model == CONFIG.embed_model) || return nothing
    DOC_INDEX[] = idx
    debug_status(io, "loaded cached doc index ($(length(idx.chunks)) chunks)")
    return idx
end

function build_index!(; io::IO = CONFIG.status_io)
    isempty(CONFIG.embed_model) &&
        throw(WinkResolveError("no embedding provider configured (load an " *
                               "in-process embedder with local_embed_model!(gguf), " *
                               "set OPENAI_API_KEY, run a local Ollama server, or " *
                               "configure!(embed_model = ..., embed_api_base = ...))"))
    chunks = build_corpus()
    isempty(chunks) && throw(WinkResolveError("no documented bindings found to index"))
    status(io,
        "building doc index: $(length(chunks)) docstring chunks via $(CONFIG.embed_model) " *
        "(one-time; cached afterwards)")
    emb = embed_texts([chunk_embed_text(c) for c in chunks]; io)
    idx = DocIndex(chunks, emb, CONFIG.embed_model, modules_fingerprint())
    DOC_INDEX[] = idx
    INDEX_FAILURE[] = nothing
    save_index(idx)
    status(io, "doc index ready ($(size(emb, 2)) vectors × $(size(emb, 1)) dims)")
    return idx
end

function ensure_index!(; io::IO = CONFIG.status_io)
    DOC_INDEX[] === nothing || return DOC_INDEX[]
    INDEX_FAILURE[] === nothing || return nothing
    load_index!(; io) === nothing || return DOC_INDEX[]
    return try
        build_index!(; io)
    catch e
        e isa InterruptException && rethrow()
        INDEX_FAILURE[] = sprint(Base.showerror, e)
        nothing
    end
end

"""
    reindex!()

(Re)build the documentation search index over everything currently loaded and
persist it to the cache. Run this after loading new packages to make their
docstrings searchable; it is never rebuilt implicitly.
"""
function reindex!(; io::IO = CONFIG.status_io)
    INDEX_FAILURE[] = nothing
    DOC_INDEX[] = nothing
    try
        build_index!(; io)
    catch e
        e isa InterruptException && rethrow()
        INDEX_FAILURE[] = sprint(Base.showerror, e)
        printstyled(io, "[Wink] doc index build failed: ", INDEX_FAILURE[], "\n";
            color = :red)
    end
    return nothing
end

# ---- retrieval ---------------------------------------------------------------

function search_scores(idx::DocIndex, q::Vector{Float32}; k::Int = 8)
    scores = idx.embeddings' * q
    k = min(k, length(scores))
    top = partialsortperm(scores, 1:k; rev = true)
    io = IOBuffer()
    for i in top
        c = idx.chunks[i]
        sig = isempty(c.sig) ? "" : " — $(c.sig)"
        loc = isempty(c.file) ? "" : "  ($(c.file):$(c.line))"
        println(io, "### ", c.binding, sig, "  [score ",
            round(scores[i]; digits = 3), "]", loc)
        println(io, first(c.text, 700))
        println(io)
    end
    return String(take!(io))
end

staleness_note(idx::DocIndex) =
    idx.modules_hash == modules_fingerprint() ? "" :
    "\n(note: modules were loaded or unloaded since indexing — run Wink.reindex!() " *
    "to refresh the doc index)"

function keyword_fallback(query::AbstractString, reason::AbstractString)
    out = try
        sprint(io -> Base.Docs.apropos(io, strip(query)))
    catch
        ""
    end
    hits = filter(!isempty, split(out, '\n'))
    body = if isempty(hits)
        "(no keyword matches either)"
    else
        lines = String[]
        for h in first(hits, 15)
            doc1 = try
                String(first(split(
                    rstrip(Markdown.plain(Base.Docs.doc(resolve_docbinding(String(h))))),
                    '\n')))
            catch
                ""
            end
            push!(lines, isempty(doc1) ? String(h) : "$h — $doc1")
        end
        join(lines, "\n")
    end
    return "(semantic search unavailable: $reason. Keyword results via apropos:)\n" * body
end

"""
    search_docs(query::String, top_k::String) -> String

Semantic search over the docstrings of every module loaded in this session.
Returns the best-matching documented bindings with signatures, source
locations, and doc excerpts. `top_k` is the number of results (`""` for the
default of 8). Use this to discover functionality when you don't know exact
names; once you know the binding, prefer get_doc for the full docstring. Falls
back to keyword matching when no embedding provider is available. The first
call may take a while — the index is built lazily and then cached.
"""
tool_search_docs(query::String, top_k::String) = tool_guard("search_docs") do
    q = strip(query)
    isempty(q) && throw(WinkResolveError("query is empty"))
    k = clamp(something(tryparse(Int, strip(top_k)), 8), 1, 25)
    isempty(CONFIG.embed_model) &&
        return keyword_fallback(q,
            "no embedding provider (load one in-process with " *
            "local_embed_model!(gguf), set OPENAI_API_KEY, run a local Ollama " *
            "server, or configure!(embed_model = ..., embed_api_base = ...))")
    idx = ensure_index!()
    idx === nothing &&
        return keyword_fallback(q, something(INDEX_FAILURE[], "index build failed"))
    try
        search_scores(idx, embed_query(idx.model, q); k) * staleness_note(idx)
    catch e
        e isa InterruptException && rethrow()
        keyword_fallback(q, "query embedding failed: $(sprint(Base.showerror, e))")
    end
end

push!(TOOL_SPECS, "search_docs" => tool_search_docs)
