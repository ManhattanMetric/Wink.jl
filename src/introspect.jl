# Human-facing introspection API. The model-facing String-in/String-out
# wrappers live in tools.jl.

"""
    SourceResult(kind, code, file, line)

Result of [`source`](@ref). `kind` is `:file` when actual source text was
recovered (via CodeTracking) and `:lowered` when only lowered IR is available
(e.g. for code defined with `eval`).
"""
struct SourceResult
    kind::Symbol
    code::String
    file::String
    line::Int
end

function Base.show(io::IO, ::MIME"text/plain", s::SourceResult)
    if s.kind === :file
        println(io, "# ", s.file, ":", s.line)
    else
        println(io, "# no source file available — lowered IR (method at ", s.file, ":",
            s.line, ")")
    end
    print(io, s.code)
end

"""
    source(m::Method) -> SourceResult
    source(f, argtypes::Type{<:Tuple}) -> SourceResult

Recover the source code of a method. Reads the actual text from the source file
via CodeTracking (Revise-aware locations); when no source file exists, falls
back to pretty-printed lowered IR.
"""
function source(m::Method)
    def = try
        CodeTracking.definition(String, m)
    catch
        nothing
    end
    if def !== nothing
        code, line = def
        file = try
            first(CodeTracking.whereis(m))
        catch
            string(m.file)
        end
        return SourceResult(:file, code, file, line)
    end
    ci = try
        Base.uncompressed_ir(m)
    catch e
        throw(WinkResolveError("no source text or lowered IR is available for `$m`: " *
                               sprint(showerror, e)))
    end
    return SourceResult(:lowered, sprint(show, MIME"text/plain"(), ci), string(m.file),
        Int(m.line))
end
source(f, argtypes::Type{<:Tuple}) = source(which(f, argtypes))

const IR_LEVELS = (:lowered, :typed, :warntype, :llvm, :native)

"""
    ir_text(f, argtypes::Type{<:Tuple}, level::Symbol) -> String

Render compiler IR for `f` at `argtypes`. `level` is one of
`$(join(string.(IR_LEVELS), ", "))`.
"""
function ir_text(f, argtypes::Type{<:Tuple}, level::Symbol)
    level in IR_LEVELS ||
        throw(WinkResolveError("unknown IR level `$level`; expected one of " *
                               join(string.(IR_LEVELS), ", ")))
    if level === :lowered
        cis = code_lowered(f, argtypes)
        isempty(cis) && throw(WinkResolveError("no lowered code found for this signature"))
        return join((sprint(show, MIME"text/plain"(), ci) for ci in cis), "\n\n")
    elseif level === :typed
        res = code_typed(f, argtypes)
        isempty(res) && throw(WinkResolveError("no typed code found for this signature"))
        return join((sprint(show, MIME"text/plain"(), p) for p in res), "\n\n")
    elseif level === :warntype
        return sprint(io -> InteractiveUtils.code_warntype(io, f, argtypes))
    elseif level === :llvm
        return sprint(io -> InteractiveUtils.code_llvm(io, f, argtypes))
    else
        return sprint(io -> InteractiveUtils.code_native(io, f, argtypes))
    end
end

"""
    docstring(name::AbstractString) -> Markdown.MD

Fetch documentation for a binding named by `name` (see
[`resolve_docbinding`](@ref) for accepted forms).
"""
docstring(name::AbstractString) = Base.Docs.doc(resolve_docbinding(name))

"""
    typeinfo_text(T::Type) -> String

Describe a type: kind, mutability, parameters, fields, supertype chain, and
direct subtypes.
"""
function typeinfo_text(T::Type)
    io = IOBuffer()
    println(io, "Type: ", T)
    base = Base.unwrap_unionall(T)
    if T isa Union
        println(io, "  kind: Union with members ", join(string.(Base.uniontypes(T)), ", "))
    elseif base isa DataType
        kind = isabstracttype(T) ? "abstract type" :
               isprimitivetype(base) ? "primitive type" :
               ismutabletype(base) ? "mutable struct" : "struct"
        println(io, "  kind: ", kind)
        if isconcretetype(T) && isbitstype(T)
            println(io, "  isbits: true, sizeof: ", sizeof(T), " bytes")
        end
        isempty(base.parameters) ||
            println(io, "  parameters: ", join(string.(base.parameters), ", "))
        if isstructtype(base) && !isabstracttype(base)
            fns = fieldnames(base)
            fts = fieldtypes(base)
            if isempty(fns)
                println(io, "  fields: (none)")
            else
                println(io, "  fields:")
                for (n, t) in zip(fns, fts)
                    println(io, "    ", n, " :: ", t)
                end
            end
        end
    end
    if !(T isa Union)
        chain = String[string(T)]
        S = T
        while S !== Any
            S = supertype(S)
            push!(chain, string(S))
        end
        println(io, "  supertype chain: ", join(chain, " <: "))
    end
    subs = try
        InteractiveUtils.subtypes(T)
    catch
        Type[]
    end
    if !isempty(subs)
        shown = join(string.(first(subs, 40)), ", ")
        extra = length(subs) > 40 ? ", …" : ""
        println(io, "  direct subtypes (", length(subs), "): ", shown, extra)
    end
    return String(take!(io))
end

"""
    modinfo_text(m::Module; all::Bool = false) -> String

List a module's contents grouped by kind. With `all = true`, unexported names
are included (compiler-generated names are always filtered out).
"""
function modinfo_text(m::Module; all::Bool = false)
    ns = filter(n -> !startswith(string(n), "#"), names(m; all = all))
    sort!(ns; by = n -> lowercase(string(n)))
    mods, types, funcs, macros, consts = Symbol[], Symbol[], Symbol[], Symbol[], Symbol[]
    for n in ns
        if startswith(string(n), "@")
            push!(macros, n)
            continue
        end
        isdefined(m, n) || continue
        v = try
            getglobal(m, n)
        catch
            continue
        end
        v === m && continue
        if v isa Module
            push!(mods, n)
        elseif v isa Type
            push!(types, n)
        elseif v isa Function
            push!(funcs, n)
        else
            push!(consts, n)
        end
    end
    io = IOBuffer()
    println(io, "Module ", m, all ? " (all names):" : " (exported names):")
    for (label, group) in ("submodules" => mods, "types" => types,
        "functions" => funcs, "macros" => macros, "other constants/globals" => consts)
        isempty(group) && continue
        shown = join(string.(first(group, 120)), ", ")
        extra = length(group) > 120 ? ", …" : ""
        println(io, "  ", label, " (", length(group), "): ", shown, extra)
    end
    return String(take!(io))
end

"""
    methodtable_text(f; argtypes = nothing) -> String

Format the method table of `f` (optionally narrowed to methods applicable to
`argtypes`) with Revise-aware source locations.
"""
function methodtable_text(f; argtypes::Union{Nothing, Type{<:Tuple}} = nothing)
    ms = argtypes === nothing ? methods(f) : methods(f, argtypes)
    n = length(ms)
    fname = try
        string(nameof(f))
    catch
        string(f)
    end
    io = IOBuffer()
    println(io, n, " method", n == 1 ? "" : "s", " for `", fname, "`:")
    for (i, m) in enumerate(ms)
        if i > 60
            println(io, "  … and ", n - 60, " more")
            break
        end
        file, line = try
            CodeTracking.whereis(m)
        catch
            (string(m.file), Int(m.line))
        end
        sig = first(split(sprint(show, m), " @ "))
        println(io, "  [", i, "] ", sig, "  @ ", file, ":", line)
    end
    return String(take!(io))
end
