# Offline package-registry search over the registries Pkg keeps on disk, plus
# an optional richer index of package descriptions/topics fetched once from
# GitHub (index_packages!) and embedded for semantic search when an embedding
# model is configured. Searching never touches the network; only the explicit
# index build does.

struct PkgDesc
    name::String
    description::String
    topics::Vector{String}
end

mutable struct PkgIndex
    entries::Vector{PkgDesc}
    fetched_at::Float64                          # unix time of the GitHub fetch
    embed_model::String                          # "" => no embeddings
    embeddings::Union{Nothing, Matrix{Float32}}  # dim × n, unit-norm columns
end

const PKG_INDEX = Ref{Union{Nothing, PkgIndex}}(nothing)

pkg_index_file() = joinpath(Scratch.@get_scratch!("pkgindex"), "index-v1.jls")

function _github_token()
    t = get(ENV, "GITHUB_TOKEN", "")
    isempty(t) || return t
    gh = Sys.which("gh")
    gh === nothing && return ""
    return try
        readchomp(`$gh auth token`)
    catch
        ""
    end
end

function _github_graphql(query::AbstractString, token::AbstractString;
        retries::Int = 5, io::IO = CONFIG.status_io)
    for attempt in 0:retries
        out = IOBuffer()
        resp = Downloads.request("https://api.github.com/graphql";
            method = "POST", input = IOBuffer(JSON3.write((; query))), output = out,
            headers = ["Authorization" => "Bearer $token",
                "Content-Type" => "application/json"])
        resp.status == 200 && return get(JSON3.read(String(take!(out))), :data, nothing)
        # 403/429 without primary-quota exhaustion is GitHub's secondary rate
        # limit; honor Retry-After and back off. Anything else is fatal.
        (resp.status in (403, 429) || resp.status >= 500) && attempt < retries ||
            throw(WinkResolveError("GitHub GraphQL request failed (HTTP $(resp.status))"))
        ra = 0
        for (k, v) in resp.headers
            lowercase(String(k)) == "retry-after" &&
                (ra = something(tryparse(Int, String(v)), 0))
        end
        wait_s = clamp(max(ra, 15 * 2^attempt), 15, 180)
        status(io, "GitHub rate limited (HTTP $(resp.status)); retrying in $(wait_s)s")
        sleep(wait_s)
    end
end

# All non-_jll registry packages hosted on GitHub, as (name, owner, repo).
function github_repos_from_registry()
    pairs = Tuple{String, String, String}[]
    for reg in Pkg.Registry.reachable_registries(), e in values(reg.pkgs)
        endswith(lowercase(e.name), "_jll") && continue
        info = Pkg.Registry.registry_info(e)
        m = match(r"github\.com[:/]([^/]+)/(.+?)(?:\.git)?/?$", something(info.repo, ""))
        m === nothing && continue
        push!(pairs, (e.name, String(m.captures[1]), String(m.captures[2])))
    end
    return sort!(pairs)
end

pkgdesc_embed_text(d::PkgDesc) = string(d.name, " — ", d.description,
    isempty(d.topics) ? "" : " [" * join(d.topics, ", ") * "]")

"""
    index_packages!(; embed = !isempty(CONFIG.embed_model)) -> PkgIndex

Build the local package-description index: fetch every registered GitHub-hosted
package's short description and topics via the GitHub GraphQL API (~120 batched
requests; needs `GITHUB_TOKEN` or a logged-in `gh` CLI), then — when `embed` is
true — embed the descriptions with the configured embedding model so
`search_packages` can match by meaning as well as by keyword. The result is
persisted to a scratch cache and reused across sessions; rerun to refresh.
This is the only registry operation that touches the network.
"""
function index_packages!(; embed::Bool = !isempty(CONFIG.embed_model),
        io::IO = CONFIG.status_io, batch::Int = 100)
    token = _github_token()
    isempty(token) &&
        throw(WinkResolveError("no GitHub credentials: set GITHUB_TOKEN or log in " *
                               "with `gh auth login`"))
    pkgs = github_repos_from_registry()
    isempty(pkgs) && throw(WinkResolveError("no package registries found on disk"))
    status(io, "fetching descriptions for $(length(pkgs)) packages from GitHub " *
               "($(cld(length(pkgs), batch)) batched requests; one-time, cached)")
    entries = PkgDesc[]
    for (bi, lo) in enumerate(1:batch:length(pkgs))
        chunk = pkgs[lo:min(lo + batch - 1, end)]
        parts = String[]
        for (i, (_, owner, repo)) in enumerate(chunk)
            push!(parts,
                "r$(i): repository(owner: \"$(owner)\", name: \"$(repo)\") { " *
                "description repositoryTopics(first: 8) { nodes { topic { name } } } }")
        end
        data = _github_graphql("query { " * join(parts, " ") * " }", token; io)
        data === nothing && throw(WinkResolveError("GitHub GraphQL returned no data " *
                                                   "(rate limit or auth problem?)"))
        sleep(1.0)   # pace requests: GitHub's secondary limit dislikes bursts
        for (i, (name, _, _)) in enumerate(chunk)
            node = get(data, Symbol("r$(i)"), nothing)
            desc = node === nothing ? "" : string(something(node.description, ""))
            topics = node === nothing ? String[] :
                     String[lowercase(string(t.topic.name)) for t in
                            node.repositoryTopics.nodes]
            push!(entries, PkgDesc(name, desc, topics))
        end
        bi % 10 == 0 &&
            status(io, "fetched $(min(lo + batch - 1, length(pkgs)))/$(length(pkgs))")
    end
    idx = PkgIndex(entries, time(), "", nothing)
    if embed
        isempty(CONFIG.embed_model) &&
            throw(WinkResolveError("embed = true but no embedding model configured"))
        idx.embeddings = embed_texts([pkgdesc_embed_text(d) for d in entries]; io)
        idx.embed_model = CONFIG.embed_model
    end
    PKG_INDEX[] = idx
    try
        Serialization.serialize(pkg_index_file(), idx)
    catch e
        @debug "Wink: could not persist the package index" exception = e
    end
    status(io, "package index ready: $(length(entries)) descriptions" *
               (embed ? ", embedded via $(idx.embed_model)" : " (keyword only)"))
    return idx
end

# Lazy load from the scratch cache; never builds (building needs the network).
function ensure_pkg_index()
    PKG_INDEX[] === nothing || return PKG_INDEX[]
    f = pkg_index_file()
    isfile(f) || return nothing
    idx = try
        Serialization.deserialize(f)
    catch
        return nothing
    end
    idx isa PkgIndex || return nothing
    PKG_INDEX[] = idx
    return idx
end

function project_dep_names()
    proj = Base.active_project()
    proj === nothing && return Set{String}()
    try
        Set{String}(keys(get(Pkg.TOML.parsefile(proj), "deps", Dict{String, Any}())))
    catch
        Set{String}()
    end
end

loaded_root_names() = Set{String}(k.name for k in keys(Base.loaded_modules))

"""
    search_packages_text(pattern; limit = 40) -> String

Layered offline package search over every registry Pkg can reach on disk.
Always matches names (case-insensitive substring; exact, then prefix, then the
rest). With the description index built ([`index_packages!`](@ref)) it also
matches descriptions and topics by keyword, and — when the index was embedded —
by meaning. `_jll` binary wrappers are excluded from name matching unless the
pattern itself mentions "jll". Each hit shows the latest registered version,
the description when known, the repository URL, and whether it is already a
dependency of the active project or loaded in this session.
"""
function search_packages_text(pattern::AbstractString; limit::Integer = 40)
    q = lowercase(strip(pattern))
    isempty(q) && throw(WinkResolveError("pattern is empty"))
    regs = Pkg.Registry.reachable_registries()
    isempty(regs) &&
        return "No package registries found on disk (run Pkg.Registry.add())."
    regnames = join((r.name for r in regs), ", ")
    entries = Dict{String, Pkg.Registry.PkgEntry}()
    for reg in regs, e in values(reg.pkgs)
        entries[e.name] = e
    end
    # Rank layers: 0 exact name, 1 name prefix, 2 name substring, 3 description
    # keyword, 4 topic keyword, 5 semantic. A package keeps its best rank.
    ranked = Dict{String, Int}()
    consider(name, rank) = haskey(entries, name) &&
        (ranked[name] = min(get(ranked, name, typemax(Int)), rank); true)
    for name in keys(entries)
        ln = lowercase(name)
        occursin(q, ln) || continue
        endswith(ln, "_jll") && !occursin("jll", q) && continue
        consider(name, ln == q ? 0 : startswith(ln, q) ? 1 : 2)
    end
    idx = ensure_pkg_index()
    descmap = Dict{String, PkgDesc}()
    if idx !== nothing
        for d in idx.entries
            descmap[d.name] = d
            if occursin(q, lowercase(d.description))
                consider(d.name, 3)
            elseif any(t -> occursin(q, t), d.topics)
                consider(d.name, 4)
            end
        end
        if idx.embeddings !== nothing && !isempty(idx.embed_model)
            try
                qv = embed_query(idx.embed_model, strip(pattern))
                scores = idx.embeddings' * qv
                for i in partialsortperm(scores, 1:min(8, length(scores)); rev = true)
                    scores[i] < 0.5 && break
                    consider(idx.entries[i].name, 5)
                end
            catch e
                e isa InterruptException && rethrow()
                @debug "Wink: semantic package search failed" exception = e
            end
        end
    end
    isempty(ranked) &&
        return "(no packages matching \"$q\" in registries: $regnames" *
               (idx === nothing ?
                "; note: only names were searched — run Wink.index_packages!() to " *
                "enable description search)" : ")")
    names = sort!(collect(keys(ranked)); by = n -> (ranked[n], lowercase(n)))
    deps = project_dep_names()
    loaded = loaded_root_names()
    io = IOBuffer()
    println(io, length(names), " package", length(names) == 1 ? "" : "s",
        " matching \"", q, "\" in registries: ", regnames,
        idx === nothing ?
        "  (names only — Wink.index_packages!() enables description search)" : "")
    for (i, name) in enumerate(names)
        if i > limit
            println(io, "  … and ", length(names) - limit, " more (narrow the pattern)")
            break
        end
        info = Pkg.Registry.registry_info(entries[name])
        vs = [v for (v, vi) in info.version_info if !vi.yanked]
        ver = isempty(vs) ? "(all versions yanked)" : "v" * string(maximum(vs))
        desc = get(descmap, name, nothing)
        blurb = desc === nothing || isempty(desc.description) ? "" :
                " — " * first(desc.description, 100)
        flags = String[]
        name in deps && push!(flags, "in project")
        name in loaded && push!(flags, "loaded")
        suffix = isempty(flags) ? "" : "  [" * join(flags, ", ") * "]"
        println(io, "  ", name, " ", ver, blurb, " — ",
            something(info.repo, "(no repo)"), suffix)
    end
    return String(take!(io))
end
