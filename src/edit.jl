# Confirmation-gated, exact-string file editing with live Revise pickup.
#
# Exact-unique-string replacement (rather than line ranges) because line
# numbers drift in a live Revise session: a stale old_string fails loudly
# instead of splicing at the wrong place, and the uniqueness check makes silent
# double-application impossible.

# Extra allowed roots, used by the test suite.
const EXTRA_EDIT_ROOTS = String[]

# Editable: the active project and packages loaded for development. Explicitly
# NOT editable: stdlib and the content-addressed package depot copies.
function edit_roots()
    roots = String[]
    proj = Base.active_project()
    proj === nothing || push!(roots, dirname(realpath_or(proj)))
    stddir = realpath_or(joinpath(Sys.BINDIR, "..", "share", "julia"))
    depot_pkgs = [realpath_or(joinpath(d, "packages")) for d in DEPOT_PATH]
    for m in values(Base.loaded_modules)
        d = pkgdir(m)
        d === nothing && continue
        rd = realpath_or(d)
        under(rd, stddir) && continue
        any(Base.Fix1(under, rd), depot_pkgs) && continue
        push!(roots, rd)
    end
    append!(roots, EXTRA_EDIT_ROOTS)
    return unique(filter(isdir, roots))
end

function editable_path(path::AbstractString)
    isempty(strip(path)) && throw(WinkResolveError("path is empty"))
    p = try
        realpath(abspath(expanduser(strip(path))))
    catch
        throw(WinkResolveError("file not found: $path (edit_file cannot create files)"))
    end
    isfile(p) || throw(WinkResolveError("not a file: $p"))
    endswith(p, ".jl") || throw(WinkResolveError("only .jl files can be edited"))
    any(Base.Fix1(under, p), edit_roots()) ||
        throw(WinkResolveError("`$p` is outside the editable perimeter (the active " *
                               "project and packages under development). Stdlib and " *
                               "installed depot packages are read-only."))
    return p
end

function backup_file(p::AbstractString)
    dir = Scratch.@get_scratch!("backups")
    dest = joinpath(dir, string(round(Int, time() * 1000), "-", basename(p)))
    cp(p, dest; force = true)
    return dest
end

"""
    backups() -> Vector{String}

Paths of the pre-edit backup copies Wink has saved (oldest first). Git is the
real recovery story; these are a convenience net for non-tracked files.
"""
backups() = sort!(readdir(Scratch.@get_scratch!("backups"); join = true))

# Diff-style preview shown at the confirmation gate: context lines around the
# match, `-` lines for the replaced text, `+` lines for the replacement.
function edit_preview(path::AbstractString, content::AbstractString,
        old_string::AbstractString, new_string::AbstractString; context::Int = 3)
    r = findfirst(old_string, content)
    startline = 1 + count(==(UInt8('\n')), @view codeunits(content)[1:(first(r) - 1)])
    all_lines = split(content, '\n')
    old_lines = split(old_string, '\n')
    lo = max(1, startline - context)
    hi = min(length(all_lines), startline + length(old_lines) - 1 + context)
    io = IOBuffer()
    println(io, path, " (line ", startline, "):")
    for i in lo:(startline - 1)
        println(io, "   ", all_lines[i])
    end
    for l in old_lines
        printstyled(io, " - ", l, "\n")
    end
    for l in split(new_string, '\n')
        printstyled(io, " + ", l, "\n")
    end
    for i in (startline + length(old_lines)):hi
        println(io, "   ", all_lines[i])
    end
    return String(take!(io))
end

"""
    edit_file(path::String, old_string::String, new_string::String) -> String

Replace exactly one occurrence of `old_string` with `new_string` in a source
file on disk; Revise then applies the change to the running session. The match
must be unique in the file — include enough surrounding lines to disambiguate —
and must be copied exactly (whitespace included) from `read_file` or
`get_source` output taken in this conversation; never reconstruct it from
memory. Editing is limited to the active project and packages loaded for
development; stdlib and installed depot packages are read-only. A backup is
saved before writing. Subject to user confirmation unless auto-approval is on.
"""
tool_edit_file(path::String, old_string::String, new_string::String) =
    tool_guard("edit_file") do
        p = editable_path(path)
        isempty(old_string) && throw(WinkResolveError("old_string is empty"))
        old_string == new_string &&
            throw(WinkResolveError("old_string and new_string are identical"))
        content = read(p, String)
        n = length(findall(old_string, content; overlap = false))
        n == 0 &&
            return "old_string was not found in $p. The file may have changed — " *
                   "re-read it with read_file and copy the exact current text."
        n > 1 &&
            return "old_string occurs $n times in $p; include more surrounding " *
                   "context to make it unique."
        preview = edit_preview(p, content, old_string, new_string)
        CONFIG.autoeval || CONFIG.confirm(:edit, preview) || return DECLINED_MSG
        backup_file(p)
        write(p, replace(content, old_string => new_string; count = 1))
        # Give the file watcher a beat, then drain Revise deterministically.
        sleep(0.25)
        err = try
            Revise.revise(; throw = true)
            nothing
        catch e
            e isa InterruptException && rethrow()
            e
        end
        err === nothing ?
        "OK: edited $p (1 replacement). Revise applied the change to the running session." :
        "EDITED $p (the change is on disk), but Revise reported: " *
        sprint(Base.showerror, err) *
        " — it will retry on the next evaluation; if this is a syntax error, fix the file."
    end

push!(TOOL_SPECS, "edit_file" => tool_edit_file)

# ---- file creation -----------------------------------------------------------

# Perimeter check for a path that does not exist yet: resolve symlinks through
# the nearest existing ancestor, then re-append the uncreated remainder, so a
# not-yet-created path cannot dodge the check that edit_file's realpath does.
function writable_new_path(path::AbstractString)
    isempty(strip(path)) && throw(WinkResolveError("path is empty"))
    p = abspath(expanduser(strip(path)))
    ispath(p) &&
        throw(WinkResolveError("`$p` already exists — write_file only creates " *
                               "new files; read_file it and use edit_file to change it"))
    any(endswith(p, ext) for ext in TEXT_FILE_EXTS) ||
        throw(WinkResolveError("only these file types can be written: " *
                               join(TEXT_FILE_EXTS, ", ")))
    anc, rest = dirname(p), basename(p)
    while !isdir(anc)
        parent = dirname(anc)
        parent == anc && break
        rest = joinpath(basename(anc), rest)
        anc = parent
    end
    resolved = joinpath(realpath_or(anc), rest)
    any(Base.Fix1(under, resolved), edit_roots()) ||
        throw(WinkResolveError("`$p` is outside the editable perimeter (the " *
                               "active project and packages under development)"))
    return resolved
end

"""
    write_file(path::String, content::String) -> String

Create a NEW text file with the given content — the way to add source files,
templates, and assets to the project. It refuses to overwrite: if the file
already exists, read_file it and use edit_file to change it. Parent
directories are created as needed. Writing is limited to the active project
and packages loaded for development, and to plain-text types (.jl, .toml,
.md, .txt, .html, .css, .js, .json). Never create files with open/write
inside eval_code — file changes must go through this tool so the user sees
the full content at the confirmation gate. Creating a file does not load it:
if the session should use its code, include it (or `using`/`import`) in a
separate step.
"""
tool_write_file(path::String, content::String) = tool_guard("write_file") do
    isempty(strip(content)) && throw(WinkResolveError("content is empty"))
    p = writable_new_path(path)
    CONFIG.autoeval || CONFIG.confirm(:write, "new file: $p\n\n" * content) ||
        return DECLINED_MSG
    mkpath(dirname(p))
    write(p, content)
    nlines = count(==('\n'), content) + (endswith(content, '\n') ? 0 : 1)
    "OK: created $p ($nlines lines). The file is not loaded into the session; " *
    "include it or extend the project's module if its code should run."
end

push!(TOOL_SPECS, "write_file" => tool_write_file)
