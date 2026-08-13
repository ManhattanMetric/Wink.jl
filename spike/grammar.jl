# Tool-call grammar: compile tool schemas into GBNF targeting gemma-4's
# canonical call syntax, and parse the calls back out of generated text.
#
#   <|tool_call>call:NAME{key:value,…}<tool_call|>
#
# with keys in dictsort order (case-insensitive — matching how render_chat
# renders historical calls, so generated and re-rendered calls agree), string
# values wrapped in the dedicated quote token <|"|>, numbers and booleans
# bare. Attached via llama_sampler_init_grammar (forced: the model MUST call)
# or llama_sampler_init_grammar_lazy (free text until it emits the trigger
# "<|tool_call>", then clamped to validity). Constrained decoding makes a
# malformed tool call unrepresentable — the sampler masks every token that
# would break the grammar.
#
# Known limitation, accepted for the spike: string contents containing the
# literal character pair "<|" can confuse the string rule's char classes.
#
# Tools use the same schema Dicts as render_chat:
#   Dict("function" => Dict("name" => …, "description" => …,
#        "parameters" => Dict("properties" => Dict(name => Dict("type" => …)),
#                             "required" => [...])))

# GBNF rule names allow only [a-zA-Z0-9-].
gbnf_rule_name(toolname) = "tool-" * replace(String(toolname), r"[^a-zA-Z0-9]" => "-")

const GBNF_QUOTE = "\"<|\\\"|>\""   # the <|"|> quote token as a GBNF literal

function gbnf_value_rule(schema_type)
    t = lowercase(String(something(schema_type, "string")))
    t == "integer" && return "integer"
    t == "number" && return "number"
    t == "boolean" && return "boolean"
    return "string"
end

"""
    tool_call_grammar(tools) -> String

Compile tool schemas into a GBNF grammar whose `root` derives exactly one
well-formed gemma-4 tool call for one of the given tools, with every
argument present in dictsort order.
"""
function tool_call_grammar(tools)
    isempty(tools) && error("no tools to compile")
    io = IOBuffer()
    names = String[]
    for tool in tools
        fn = something(field(tool, :function), tool)
        name = String(field(fn, :name))
        rule = gbnf_rule_name(name)
        push!(names, rule)
        params = something(field(fn, :parameters), Dict{String, Any}())
        props = something(field(params, :properties), Dict{String, Any}())
        args = String[]
        for key in g4_sortkeys(props)
            push!(args, "\"$key:\" " * gbnf_value_rule(field(props[key], :type)))
        end
        argbody = isempty(args) ? "" : join(args, " \",\" ")
        println(io, rule, " ::= \"", name, "{\" ", argbody, " \"}\"")
    end
    println(io, "call ::= ", join(names, " | "))
    println(io, "root ::= \"<|tool_call>call:\" call \"<tool_call|>\"")
    println(io, "string ::= ", GBNF_QUOTE, " strchar* ", GBNF_QUOTE)
    println(io, "strchar ::= [^<] | \"<\" [^|]")
    println(io, "integer ::= \"-\"? [0-9]+")
    println(io, "number ::= \"-\"? [0-9]+ (\".\" [0-9]+)?")
    println(io, "boolean ::= \"true\" | \"false\"")
    return String(take!(io))
end

# ---- parsing generated calls --------------------------------------------------

const QUOTE_TOKEN = "<|\"|>"

# Parse one value of the canonical argument syntax starting at byte index i.
function _parse_value(s::AbstractString, i::Int)
    if startswith(SubString(s, i), QUOTE_TOKEN)
        i += ncodeunits(QUOTE_TOKEN)
        j = findnext(QUOTE_TOKEN, s, i)
        j === nothing && error("unterminated string in tool call")
        return String(SubString(s, i, prevind(s, first(j)))), first(j) + ncodeunits(QUOTE_TOKEN)
    elseif startswith(SubString(s, i), "true")
        return true, i + 4
    elseif startswith(SubString(s, i), "false")
        return false, i + 5
    elseif startswith(SubString(s, i), "null")
        return nothing, i + 4
    elseif s[i] == '{'
        d = Dict{String, Any}()
        i += 1
        s[i] == '}' && return d, i + 1
        while true
            j = findnext(':', s, i)
            k = String(SubString(s, i, prevind(s, j)))
            v, i = _parse_value(s, j + 1)
            d[k] = v
            s[i] == ',' || break
            i += 1
        end
        s[i] == '}' || error("expected } in tool-call arguments")
        return d, i + 1
    elseif s[i] == '['
        a = Any[]
        i += 1
        s[i] == ']' && return a, i + 1
        while true
            v, i = _parse_value(s, i)
            push!(a, v)
            s[i] == ',' || break
            i += 1
        end
        s[i] == ']' || error("expected ] in tool-call arguments")
        return a, i + 1
    else
        j = something(findnext(c -> c in (',', '}', ']'), s, i), lastindex(s) + 1) - 1
        tok = SubString(s, i, j)
        v = something(tryparse(Int, tok), tryparse(Float64, tok), String(tok))
        return v, j + 1
    end
end

"""
    parse_tool_call(text) -> (; name, args) or nothing

Extract and parse the first `<|tool_call>call:…<tool_call|>` block from
generated text. `args` is a `Dict{String,Any}`.
"""
function parse_tool_call(text::AbstractString)
    m = match(r"<\|tool_call>call:([A-Za-z0-9_]+)\{"s, text)
    m === nothing && return nothing
    body_start = m.offset + ncodeunits(m.match) - 1   # at the '{'
    args, _ = _parse_value(text, body_start)
    return (name = String(m.captures[1]), args = args)
end
