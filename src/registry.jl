# Offline package-registry search over the registries Pkg keeps on disk.

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

Case-insensitive substring search over package names in every registry Pkg can
reach on disk — no network. `_jll` binary wrappers are excluded unless the
pattern itself mentions "jll". Exact matches rank first, then prefix matches,
then the rest alphabetically; each hit shows the latest registered version, the
repository URL, and whether it is already a dependency of the active project or
loaded in this session.
"""
function search_packages_text(pattern::AbstractString; limit::Integer = 40)
    q = lowercase(strip(pattern))
    isempty(q) && throw(WinkResolveError("pattern is empty"))
    regs = Pkg.Registry.reachable_registries()
    isempty(regs) &&
        return "No package registries found on disk (run Pkg.Registry.add())."
    regnames = join((r.name for r in regs), ", ")
    hits = Pkg.Registry.PkgEntry[]
    for reg in regs, entry in values(reg.pkgs)
        name = lowercase(entry.name)
        occursin(q, name) || continue
        endswith(name, "_jll") && !occursin("jll", q) && continue
        push!(hits, entry)
    end
    isempty(hits) && return "(no packages matching \"$q\" in registries: $regnames)"
    sort!(hits,
        by = e -> (lowercase(e.name) != q, !startswith(lowercase(e.name), q),
            lowercase(e.name)))
    deps = project_dep_names()
    loaded = loaded_root_names()
    io = IOBuffer()
    println(io, length(hits), " package", length(hits) == 1 ? "" : "s",
        " matching \"", q, "\" in registries: ", regnames)
    for (i, e) in enumerate(hits)
        if i > limit
            println(io, "  … and ", length(hits) - limit, " more (narrow the pattern)")
            break
        end
        info = Pkg.Registry.registry_info(e)
        vs = [v for (v, vi) in info.version_info if !vi.yanked]
        ver = isempty(vs) ? "(all versions yanked)" : "v" * string(maximum(vs))
        flags = String[]
        e.name in deps && push!(flags, "in project")
        e.name in loaded && push!(flags, "loaded")
        suffix = isempty(flags) ? "" : "  [" * join(flags, ", ") * "]"
        println(io, "  ", e.name, " ", ver, " — ", something(info.repo, "(no repo)"),
            suffix)
    end
    return String(take!(io))
end
