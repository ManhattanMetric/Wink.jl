# Evaluation of model-written code in the user's Main, with REPL semantics.

"""
    EvalResult(ok, value_repr, output, error_text)

Outcome of [`eval_in_main`](@ref): success flag, the `text/plain` rendering of
the final value, captured stdout/stderr, and a formatted error + trimmed
backtrace when evaluation failed.
"""
struct EvalResult
    ok::Bool
    value_repr::String
    output::String
    error_text::String
end

function render_value(v)
    v === nothing && return "nothing"
    return try
        Base.invokelatest(repr, MIME("text/plain"), v; context = :limit => true)
    catch
        try
            Base.invokelatest(repr, v)
        catch
            "<unshowable value of type $(typeof(v))>"
        end
    end
end

function format_error(err, bt)
    msg = try
        sprint(Base.showerror, err)
    catch
        "error of type $(typeof(err))"
    end
    frames = String[]
    try
        for fr in Base.stacktrace(bt)
            s = string(fr)
            (occursin("IOCapture", s) || occursin("eval_in_main", s)) && continue
            fr.func === :eval && occursin("boot.jl", string(fr.file)) && break
            push!(frames, s)
            length(frames) >= 15 && break
        end
    catch
    end
    return isempty(frames) ? "ERROR: $msg" :
           "ERROR: $msg\nStacktrace (trimmed):\n  " * join(frames, "\n  ")
end

"""
    eval_in_main(code::AbstractString; softscope = true) -> EvalResult

Parse `code` in full (nothing executes if any part fails to parse), then
evaluate each top-level expression in `Main` with REPL soft-scope semantics,
capturing stdout/stderr. Ctrl-C propagates; all other errors are captured and
formatted.
"""
function eval_in_main(code::AbstractString; softscope::Bool = true)
    exprs = try
        Meta.parseall(String(code); filename = "wink-eval")
    catch e
        return EvalResult(false, "", "", "parse error: $(sprint(showerror, e))")
    end
    for ex in exprs.args
        if ex isa Expr && (ex.head === :error || ex.head === :incomplete)
            return EvalResult(false, "", "",
                "parse error: " * join(string.(ex.args), " "))
        end
    end
    cap = IOCapture.capture(; rethrow = InterruptException, color = false) do
        local result = nothing
        for ex in exprs.args
            ex isa LineNumberNode && continue
            result = Core.eval(Main, softscope ? REPL.softscope(ex) : ex)
        end
        result
    end
    cap.error && return EvalResult(false, "", cap.output,
        format_error(cap.value, cap.backtrace))
    return EvalResult(true, render_value(cap.value), cap.output, "")
end

"""
    format_for_model(r::EvalResult; limit = CONFIG.tool_output_limit) -> String

Render an [`EvalResult`](@ref) as the tool-result string the model sees.
"""
function format_for_model(r::EvalResult; limit::Integer = CONFIG.tool_output_limit)
    parts = String[]
    push!(parts,
        r.ok ? "value: " * truncate_output(r.value_repr; limit) :
        truncate_output(r.error_text; limit))
    isempty(strip(r.output)) ||
        push!(parts, "stdout/stderr:\n" * truncate_output(r.output; limit))
    return join(parts, "\n\n")
end

"""
    eval_code(code::String) -> String

Evaluate Julia code directly in the user's live `Main` session and return the
resulting value plus captured stdout/stderr. The code must be syntactically
valid Julia — not Python: blocks close with `end` (never a `:` after `for`/
`if`), indexing is 1-based, and only functions that exist in this session may
be called (verify with list_names or get_doc when unsure). It runs with REPL
semantics (soft scope), so write it exactly as you would type it at the `julia>` prompt;
definitions, variables, and side effects persist for both you and the user.
Unless auto-approval is on, the user is shown the code first and may decline —
a DECLINED result means do not retry verbatim; explain your intent or propose a
revision. Prefer small, verifiable snippets over one big blob, and check
intermediate results as you go.
"""
tool_eval_code(code::String) = tool_guard("eval_code") do
    isempty(strip(code)) && throw(WinkResolveError("code is empty"))
    CONFIG.autoeval || CONFIG.confirm(:eval, code) || return DECLINED_MSG
    format_for_model(eval_in_main(code))
end

push!(TOOL_SPECS, "eval_code" => tool_eval_code)
