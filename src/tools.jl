# Model-facing tools.
#
# PromptingTools derives each tool's JSON schema from the function signature and
# uses the docstring as the model-facing description, which imposes three rules
# on every `tool_*` function here: exactly one method, all-positional arguments,
# every argument `::String` (the empty string is the "not given" sentinel).
# Tools are registered under clean names (without the `tool_` prefix) via
# `tool_call_signature`'s `name` keyword; passing the pre-built `Tool` objects
# to `aitools` bypasses the Anthropic path's 100-character description
# truncation.

const DECLINED_MSG = "DECLINED: the user chose not to run this. Do not retry the " *
                     "identical code/edit; explain what you wanted to do, or propose " *
                     "a revised version."

"""
    truncate_output(s; limit = CONFIG.tool_output_limit) -> String

Cap tool output at `limit` characters, keeping the head and the last 200
characters with an explicit truncation marker in between.
"""
function truncate_output(s::AbstractString; limit::Integer = CONFIG.tool_output_limit)
    length(s) <= limit && return String(s)
    tail = min(200, max(limit ÷ 4, 1))
    head = max(limit - tail, 1)
    dropped = length(s) - head - tail
    return string(first(s, head), "\n… [truncated ", dropped, " characters] …\n",
        last(s, tail))
end

# Run a tool body, converting any exception into a model-readable "ERROR: ..."
# string (tools must never throw into the agent loop).
function tool_guard(f::Function, name::AbstractString)
    out = try
        f()
    catch e
        e isa InterruptException && rethrow()
        msg = e isa WinkResolveError ? e.msg : sprint(showerror, e)
        "ERROR in $name: $msg"
    end
    return truncate_output(out)
end

# Resolve to exactly one Method, or return a String explaining what to do next.
function single_method_or_table(signature::AbstractString)
    rc = resolve_call(signature)
    rc.method === nothing || return rc.method
    if rc.argtypes === nothing
        ms = methods(rc.func)
        length(ms) == 1 && return first(ms)
        return "This function has $(length(ms)) methods and no argument types were " *
               "given. Pick one signature and call again with types:\n" *
               methodtable_text(rc.func)
    end
    return "No method matches those argument types. Existing methods:\n" *
           methodtable_text(rc.func)
end

"""
    list_methods(signature::String) -> String

List the methods of a function with their signatures and current source
locations. `signature` may be a bare or module-qualified function name
(`"sort"`, `"Base.sort"`) or include argument types (`"sort(::Vector{Int})"`)
to narrow the listing to applicable methods. Call this before `get_source`
whenever you are unsure which method dispatch would select.
"""
tool_list_methods(signature::String) = tool_guard("list_methods") do
    rc = resolve_call(signature)
    methodtable_text(rc.func; argtypes = rc.argtypes)
end

"""
    methods_with(type_name::String, module_name::String, supertypes::String) -> String

List the methods that accept an argument of the given type — the reverse of
`list_methods`, and the direct answer to "what can I call with an X?" or "how
many methods take an X?". Searches the exported names of every loaded module,
or — when `module_name` is given (`""` for all) — that module including its
unexported functions. Pass `"true"` for `supertypes` to also count methods
declared for supertypes (e.g. `methods_with("String", "", "true")` includes
`::AbstractString` methods). The total count is the first line; the listing
after it may be truncated.
"""
tool_methods_with(type_name::String, module_name::String, supertypes::String) =
    tool_guard("methods_with") do
        T = resolve_type(type_name)
        mod = if isempty(strip(module_name))
            nothing
        else
            m = resolve_binding(module_name)
            m isa Module ||
                throw(WinkResolveError("`$module_name` is a $(typeof(m)), not a module"))
            m
        end
        methodswith_text(T; mod, supertypes = lowercase(strip(supertypes)) == "true")
    end

"""
    get_source(signature::String) -> String

Return the actual source code of a method, read from its source file — this is
ground truth; prefer it over recalling code from memory. `signature` should
normally include argument types (`"sort(::Vector{Int})"`,
`"TestPkg.greet(::String)"`); a bare name works when the function has exactly
one method. If no source file exists (e.g. the method was defined via `eval`),
clearly-marked lowered IR is returned instead. When several methods match, a
method table is returned so you can pick one signature and call again.
"""
tool_get_source(signature::String) = tool_guard("get_source") do
    m = single_method_or_table(signature)
    m isa String && return m
    s = source(m)
    header = s.kind === :file ? "# source: $(s.file):$(s.line)\n" :
             "# no source file available — lowered IR for `$(m)`:\n"
    header * s.code
end

"""
    get_ir(signature::String, level::String) -> String

Show compiler intermediate representation for a method. `level` is one of
"lowered", "typed", "warntype" (type-stability analysis — the best first stop
for performance questions), "llvm", or "native". The signature must include
argument types (`"twice(::Int)"`) because IR is specific to concrete argument
types.
"""
tool_get_ir(signature::String, level::String) = tool_guard("get_ir") do
    rc = resolve_call(signature)
    rc.argtypes === nothing &&
        return "get_ir requires argument types in the signature, e.g. " *
               "\"twice(::Int)\". Use list_methods to see available signatures."
    ir_text(rc.func, rc.argtypes, Symbol(lowercase(strip(level))))
end

"""
    get_doc(name::String) -> String

Fetch the docstring for a function, type, macro, constant, or module. Accepts
bare or qualified names: `"sort"`, `"Base.Docs.doc"`, `"@time"`,
`"Base.@show"`, `"LinearAlgebra"`. Returns rendered plain-text documentation,
or a note when the binding is undocumented.
"""
tool_get_doc(name::String) = tool_guard("get_doc") do
    rstrip(Markdown.plain(docstring(name)))
end

"""
    type_info(type_name::String) -> String

Describe a type: struct kind and mutability, type parameters, field names with
field types, supertype chain, direct subtypes, and memory layout when concrete.
Accepts qualified and parameterized forms: `"TestPkg.Point"`, `"Vector{Int}"`,
`"AbstractDict"`.
"""
tool_type_info(type_name::String) = tool_guard("type_info") do
    typeinfo_text(resolve_type(type_name))
end

"""
    module_info(module_name::String, include_private::String) -> String

List a module's contents grouped by kind (submodules, types, functions, macros,
constants). Pass `""` or `"false"` for `include_private` to see exported names
only, `"true"` to include unexported internals.
"""
tool_module_info(module_name::String, include_private::String) =
    tool_guard("module_info") do
        m = resolve_binding(module_name)
        m isa Module ||
            throw(WinkResolveError("`$module_name` is a $(typeof(m)), not a module"))
        modinfo_text(m; all = lowercase(strip(include_private)) == "true")
    end

"""
    list_variables(module_name::String, pattern::String) -> String

List the bindings a module currently holds with each one's size and a one-line
summary of its value (like the REPL's `varinfo`). With `module_name` `""` it
shows the user's workspace `Main` — the way to see what the user has defined
and computed in this session before reasoning about their data. `pattern`
optionally filters names by regex (`""` for all).
"""
tool_list_variables(module_name::String, pattern::String) =
    tool_guard("list_variables") do
        m = if isempty(strip(module_name))
            Main
        else
            b = resolve_binding(module_name)
            b isa Module ||
                throw(WinkResolveError("`$module_name` is a $(typeof(b)), not a module"))
            b
        end
        pat = isempty(strip(pattern)) ? r"" : Regex(strip(pattern))
        rstrip(Markdown.plain(varinfo(m, pat)))
    end

"""
    where_defined(signature::String) -> String

Return the `file:line` where a method is currently defined (Revise-aware).
Use this before `read_file`/`edit_file` to locate the code to inspect or
change. Include argument types when the function has multiple methods.
"""
tool_where_defined(signature::String) = tool_guard("where_defined") do
    m = single_method_or_table(signature)
    m isa String && return m
    file, line = CodeTracking.whereis(m)
    "$file:$line"
end

"""
    read_file(path::String, first_line::String, last_line::String) -> String

Read a source file with each line prefixed by its line number. Pass line
numbers to read an inclusive range, or `""` for both to read from the start
(output is capped). Reading is restricted to the active project, loaded
packages, and Julia's own source tree. Use after `where_defined` to see code in
context and to copy the exact text needed for `edit_file`.
"""
tool_read_file(path::String, first_line::String, last_line::String) =
    tool_guard("read_file") do
        p = readable_path(path)
        lines = readlines(p)
        isempty(lines) && return "(empty file)"
        lo = isempty(strip(first_line)) ? 1 :
             clamp(something(tryparse(Int, strip(first_line)), 1), 1, length(lines))
        hi = isempty(strip(last_line)) ? length(lines) :
             clamp(something(tryparse(Int, strip(last_line)), length(lines)), lo,
            length(lines))
        hi = min(hi, lo + 399)
        body = join(("$(lpad(i, 5)): $(lines[i])" for i in lo:hi), "\n")
        hi < length(lines) ? body * "\n… ($(length(lines) - hi) more lines)" : body
    end

"""
    list_names(pattern::String) -> String

Keyword search over the documentation of everything loaded in this session
(like the REPL's `apropos`): returns the names of bindings whose docstrings
mention `pattern`. A cheap complement to `search_docs` for quick "is there a
function for X" checks.
"""
tool_list_names(pattern::String) = tool_guard("list_names") do
    q = strip(pattern)
    isempty(q) && throw(WinkResolveError("pattern is empty"))
    out = sprint(io -> Base.Docs.apropos(io, q))
    isempty(strip(out)) ? "(no matches for \"$q\")" : out
end

"""
    search_packages(pattern::String) -> String

Search the package registries Pkg has on disk (e.g. General, ~14k packages) —
offline, no network. Use before recommending, or assuming the existence of,
any package. Names always match by case-insensitive substring; when the local
description index has been built, descriptions and topics match too (by
keyword, and by meaning if the index is embedded), so both short name guesses
("csv") and task phrases ("read parquet files") work — the output's first line
says which layers are active. Results rank exact name matches first and show
each hit's latest registered version, description, repository URL, and whether
it is already a dependency of the active project or loaded in this session.
`_jll` binary wrappers are hidden unless the pattern mentions "jll".
"""
tool_search_packages(pattern::String) = tool_guard("search_packages") do
    search_packages_text(pattern)
end

# ---- read perimeter ----------------------------------------------------------

# Plain-text file types Wink will read (read_file) or create (write_file).
# edit_file stays .jl-only: its Revise round-trip only makes sense for source.
const TEXT_FILE_EXTS = (".jl", ".toml", ".md", ".txt", ".html", ".css", ".js",
    ".json")

realpath_or(p::AbstractString) = try
    realpath(p)
catch
    abspath(p)
end

function read_roots()
    roots = String[]
    proj = Base.active_project()
    proj === nothing || push!(roots, dirname(realpath_or(proj)))
    # The session's working directory is the user's workspace even when the
    # active project points elsewhere (e.g. the default environment).
    push!(roots, realpath_or(pwd()))
    for mod in values(Base.loaded_modules)
        d = pkgdir(mod)
        d === nothing || push!(roots, realpath_or(d))
    end
    push!(roots, realpath_or(joinpath(Sys.BINDIR, "..", "share", "julia")))
    for depot in DEPOT_PATH
        push!(roots, realpath_or(joinpath(depot, "packages")))
        push!(roots, realpath_or(joinpath(depot, "dev")))
    end
    return unique(filter(isdir, roots))
end

under(p::AbstractString, root::AbstractString) = p == root ||
                                                 startswith(p, root * Base.Filesystem.path_separator)

function readable_path(path::AbstractString)
    isempty(strip(path)) && throw(WinkResolveError("path is empty"))
    p = try
        realpath(abspath(expanduser(strip(path))))
    catch
        throw(WinkResolveError("file not found: $path"))
    end
    isfile(p) || throw(WinkResolveError("not a file: $p"))
    any(Base.Fix1(under, p), read_roots()) ||
        throw(WinkResolveError("`$p` is outside the readable perimeter (the active " *
                               "project, the session's working directory, loaded " *
                               "packages, and Julia's source tree)"))
    any(endswith(p, ext) for ext in TEXT_FILE_EXTS) ||
        throw(WinkResolveError("only these file types can be read: " *
                               join(TEXT_FILE_EXTS, ", ")))
    return p
end

# ---- registry ----------------------------------------------------------------

# Ordered list of (model-facing name => implementation). Gated tools (eval_code,
# edit_file) and search_docs are appended by later milestones' files.
const TOOL_SPECS = Pair{String, Function}[
    "list_methods" => tool_list_methods,
    "methods_with" => tool_methods_with,
    "get_source" => tool_get_source,
    "get_ir" => tool_get_ir,
    "get_doc" => tool_get_doc,
    "type_info" => tool_type_info,
    "module_info" => tool_module_info,
    "list_variables" => tool_list_variables,
    "where_defined" => tool_where_defined,
    "read_file" => tool_read_file,
    "list_names" => tool_list_names,
    "search_packages" => tool_search_packages,
]

"""
    build_tool_map() -> Dict{String, PromptingTools.AbstractTool}

Build the model-facing tool registry from `TOOL_SPECS`. Descriptions come from
the tool functions' docstrings (kept intact well past the Anthropic path's
100-character default by pre-building `Tool` objects here).
"""
function build_tool_map()
    tm = Dict{String, PT.AbstractTool}()
    for (name, fn) in TOOL_SPECS
        merge!(tm, PT.tool_call_signature(fn; name, max_description_length = 4000))
    end
    return tm
end
