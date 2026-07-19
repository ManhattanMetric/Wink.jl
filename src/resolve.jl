# Resolution of model-supplied strings ("sort", "Base.sort", "sort(::Vector{Int})",
# "Vector{<:Real}") into live functions, types, methods, and doc bindings.

"""
    WinkResolveError(msg)

Raised when a model-supplied name, type, or signature string cannot be
resolved. The message is written to be shown back to the model verbatim so it
can self-correct.
"""
struct WinkResolveError <: Exception
    msg::String
end
Base.showerror(io::IO, e::WinkResolveError) = print(io, e.msg)

"""
    ResolvedCall(func, argtypes, method)

Result of [`resolve_call`](@ref): the callable, the `Tuple` type of argument
types (or `nothing` when the signature had no types), and the `Method` selected
by dispatch (or `nothing` when none matched / no types were given).
"""
struct ResolvedCall
    func::Any
    argtypes::Union{Nothing, Type{<:Tuple}}
    method::Union{Nothing, Method}
end

function parse_or_throw(str::AbstractString)
    ex = try
        Meta.parse(strip(str))
    catch e
        throw(WinkResolveError("could not parse `$str`: $(sprint(showerror, e))"))
    end
    ex isa Expr && ex.head === :incomplete &&
        throw(WinkResolveError("`$str` is an incomplete Julia expression"))
    ex === nothing && throw(WinkResolveError("`$str` is empty"))
    return ex
end

# `Base.Docs.doc` -> [:Base, :Docs, :doc]; returns `nothing` for non-path exprs.
function path_segments(ex)
    ex isa Symbol && return Symbol[ex]
    ex isa QuoteNode && ex.value isa Symbol && return Symbol[ex.value]
    if ex isa Expr && ex.head === :. && length(ex.args) == 2
        rest = ex.args[2]
        (rest isa QuoteNode && rest.value isa Symbol) || return nothing
        head = path_segments(ex.args[1])
        head === nothing && return nothing
        return push!(head, rest.value)
    end
    return nothing
end

function resolve_first(name::Symbol)
    for root in (Main, Base, Core)
        isdefined(root, name) && return getproperty(root, name)
    end
    for mod in values(Base.loaded_modules)
        nameof(mod) === name && return mod
    end
    throw(WinkResolveError("`$name` is not defined in Main, Base, or Core, and no " *
                           "loaded module is named `$name`. Check the spelling, or qualify " *
                           "the name with its owning module (e.g. `SomePkg.$name`)."))
end

function name_suggestions(m::Module, s::Symbol)
    frag = lowercase(string(s))
    cands = String[]
    for n in names(m)
        c = string(n)
        occursin(frag, lowercase(c)) && push!(cands, c)
        length(cands) >= 5 && break
    end
    isempty(cands) ? "" : " Did you mean: $(join(cands, ", "))?"
end

function resolve_segments(segs::Vector{Symbol})
    obj = resolve_first(segs[1])
    for s in @view segs[2:end]
        obj isa Module ||
            throw(WinkResolveError("cannot look up `$s` inside `$(typeof(obj))`; " *
                                   "`$(join(segs, '.'))` walks through a non-module value"))
        isdefined(obj, s) ||
            throw(WinkResolveError("`$s` is not defined in module `$obj`." *
                                   name_suggestions(obj, s)))
        obj = getproperty(obj, s)
    end
    return obj
end

"""
    resolve_binding(path::AbstractString) -> Any

Resolve a (possibly dotted) name like `"sort"`, `"Base.Docs.doc"`, or
`"Base.@show"` to the object it names. Roots are searched in `Main`, `Base`,
`Core`, then among all loaded modules by name. Throws [`WinkResolveError`](@ref)
with a model-friendly message on failure.
"""
function resolve_binding(path::AbstractString)
    ex = parse_or_throw(path)
    ex isa Expr && ex.head === :macrocall && (ex = ex.args[1])
    segs = path_segments(ex)
    segs === nothing &&
        throw(WinkResolveError("`$path` is not a simple (possibly dotted) name like " *
                               "`sort` or `Base.Docs.doc`"))
    return resolve_segments(segs)
end

"""
    resolve_docbinding(name::AbstractString) -> Base.Docs.Binding

Resolve a name to the documentation `Binding` used by the docsystem. Handles
functions, types, constants, macros (`"@time"`, `"Base.@show"`), and modules.
"""
function resolve_docbinding(name::AbstractString)
    ex = parse_or_throw(name)
    ex isa Expr && ex.head === :macrocall && (ex = ex.args[1])
    segs = path_segments(ex)
    segs === nothing &&
        throw(WinkResolveError("`$name` is not a name I can look up documentation for; " *
                               "use forms like `sort`, `Base.sort`, or `Base.@show`"))
    if length(segs) == 1
        s = only(segs)
        for root in (Main, Base, Core)
            isdefined(root, s) && return Base.Docs.Binding(root, s)
        end
        for mod in values(Base.loaded_modules)
            nameof(mod) === s && return Base.Docs.Binding(mod, s)
        end
        throw(WinkResolveError("`$s` is not defined in Main, Base, or Core, and no " *
                               "loaded module is named `$s`"))
    end
    parent = resolve_segments(segs[1:(end - 1)])
    parent isa Module ||
        throw(WinkResolveError("`$(join(segs[1:(end - 1)], '.'))` is not a module"))
    isdefined(parent, segs[end]) ||
        throw(WinkResolveError("`$(segs[end])` is not defined in module `$parent`." *
                               name_suggestions(parent, segs[end])))
    return Base.Docs.Binding(parent, segs[end])
end

# ---- guarded type-expression evaluation --------------------------------------
#
# Type strings are evaluated in Main, but only after a whitelist walk over the
# parsed expression. This blocks arbitrary code smuggled into type position
# ("typeof(run(`...`))") while allowing everything type syntax legitimately
# needs: names, dotted paths, curly braces, `where`, `<:`, tuples, number/symbol
# literals, and `typeof(<name>)`.

const _SAFE_TYPE_HEADS = (:., :curly, :where, :tuple, :(<:), :(>:))

function is_safe_type_expr(ex)
    ex isa Symbol && return true
    ex isa QuoteNode && return ex.value isa Symbol
    ex isa Number && return true
    if ex isa Expr
        if ex.head === :call
            return length(ex.args) == 2 && ex.args[1] === :typeof &&
                   path_segments(ex.args[2]) !== nothing
        end
        ex.head in _SAFE_TYPE_HEADS || return false
        return all(is_safe_type_expr, ex.args)
    end
    return false
end

# Replace dotted paths with their resolved values so that names from loaded-but-
# not-imported modules (e.g. "REPL.Terminals.TTYTerminal") evaluate in Main.
function qualify_type_expr(ex)
    if ex isa Expr
        if ex.head === :.
            segs = path_segments(ex)
            if segs !== nothing
                resolved = try
                    resolve_segments(segs)
                catch
                    nothing
                end
                resolved === nothing || return resolved
            end
            return ex
        end
        return Expr(ex.head, map(qualify_type_expr, ex.args)...)
    end
    return ex
end

function resolve_type_expr(ex, orig::AbstractString = string(ex))
    is_safe_type_expr(ex) ||
        throw(WinkResolveError("type expression `$orig` uses disallowed syntax; only type " *
                               "names (possibly module-qualified), curly braces, `where` " *
                               "clauses, tuples, `<:`, and `typeof(name)` are allowed"))
    val = try
        Core.eval(Main, qualify_type_expr(ex))
    catch e
        throw(WinkResolveError("could not evaluate type `$orig`: $(sprint(showerror, e))"))
    end
    val isa Type ||
        throw(WinkResolveError("`$orig` is not a type (it evaluated to a value of type " *
                               "$(typeof(val)))"))
    return val
end

"""
    resolve_type(str::AbstractString) -> Type

Resolve a type string like `"Vector{Int}"`, `"TestPkg.Point"`, or
`"AbstractArray{<:Real}"` to the `Type` it names, using a whitelisted
evaluation (see `is_safe_type_expr`).
"""
resolve_type(str::AbstractString) = resolve_type_expr(parse_or_throw(str), str)

"""
    resolve_call(sig::AbstractString) -> ResolvedCall

Resolve a signature string. Two forms are accepted:

  * `"sort"` / `"Base.sort"` — just the callable; `argtypes` and `method` are
    `nothing`.
  * `"sort(::Vector{Int})"` / `"combine(::Int, x::Int)"` / `"twice(Int)"` — the
    callable plus argument types; `method` is the method `which` selects, or
    `nothing` when no method matches.
"""
function resolve_call(sig::AbstractString)
    ex = parse_or_throw(sig)
    if ex isa Expr && ex.head === :call
        fsegs = path_segments(ex.args[1])
        fsegs === nothing &&
            throw(WinkResolveError("cannot resolve the callee in `$sig`; " *
                                   "use `func(...)` or `Module.func(...)`"))
        f = resolve_segments(fsegs)
        argts = Type[]
        for a in ex.args[2:end]
            a isa Expr && a.head === :parameters &&
                throw(WinkResolveError("keyword arguments do not participate in Julia " *
                                       "dispatch; drop them from `$sig`"))
            tex = a isa Expr && a.head === :(::) ?
                  (length(a.args) == 1 ? a.args[1] : a.args[2]) : a
            push!(argts, resolve_type_expr(tex, string(tex)))
        end
        tt = Tuple{argts...}
        m = try
            which(f, tt)
        catch
            nothing
        end
        return ResolvedCall(f, tt, m)
    end
    return ResolvedCall(resolve_binding(sig), nothing, nothing)
end
