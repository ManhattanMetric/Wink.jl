# The pure replacement for llama.cpp's lazy GBNF sampler: tool schemas
# compile into prefix matchers over gemma-4's canonical call syntax
#
#   <|tool_call>call:NAME{key:value,…}<tool_call|>
#
# and the sampler proposes-and-verifies — a candidate token is accepted iff
# appending its piece keeps the call region a valid prefix of some tool's
# expansion; otherwise its logit is erased and the sampler redraws. Because
# we own the logits, this needs no grammar engine: a call is a finite
# alternation of literal segments and typed value holes (string / integer /
# number / boolean, keys in dictsort order — matching tool_call_grammar and
# render_chat), so prefix validity is a deterministic byte walk.
#
# Same accepted limitation as the GBNF version: string values cannot contain
# the literal pair "<|" (the closing quote token <|"|> is recognized by it).

const CALL_TRIGGER = "<|tool_call>"
const CALL_CLOSE = "<tool_call|>"
const QUOTE_BYTES = codeunits("<|\"|>")

struct ToolMatcher
    lits::Vector{Vector{UInt8}}   # length(holes) + 1 literal segments
    holes::Vector{Symbol}         # :string | :integer | :number | :boolean
end

_value_kind(schema_type) = begin
    t = lowercase(String(something(schema_type, "string")))
    t == "integer" ? :integer :
    t == "number" ? :number :
    t == "boolean" ? :boolean : :string
end

"""
    compile_matchers(tools) -> Vector{ToolMatcher}

Compile tool schema dicts (the `_tool_schema` shape) into prefix matchers
for the canonical call body `call:NAME{key:value,…}<tool_call|>`.
"""
function compile_matchers(tools)
    map(collect(tools)) do tool
        fn = something(field(tool, :function), tool)
        name = String(field(fn, :name))
        params = something(field(fn, :parameters), Dict{String, Any}())
        props = something(field(params, :properties), Dict{String, Any}())
        lits = Vector{UInt8}[]
        holes = Symbol[]
        cur = "call:" * name * "{"
        for (n, key) in enumerate(g4_sortkeys(props))
            cur *= (n > 1 ? "," : "") * String(key) * ":"
            push!(lits, Vector{UInt8}(codeunits(cur)))
            cur = ""
            push!(holes, _value_kind(field(props[key], :type)))
        end
        cur *= "}" * CALL_CLOSE
        push!(lits, Vector{UInt8}(codeunits(cur)))
        ToolMatcher(lits, holes)
    end
end

# byte-wise match of pat at s[i:end]: 0 mismatch / 1 s ended inside pat /
# 2 full match — returning the index after the match
function _match_bytes(s::AbstractVector{UInt8}, i::Int, pat)
    for b in pat
        i > length(s) && return 1, i
        s[i] == b || return 0, i
        i += 1
    end
    return 2, i
end

# can s[i:end] be a prefix of lits[k] value_k lits[k+1] … ? 0/1/2 like above
function _seg(s, i, m::ToolMatcher, k::Int)
    c, i = _match_bytes(s, i, m.lits[k])
    c == 0 && return 0
    c == 1 && return 1
    k > length(m.holes) && return i > length(s) ? 2 : 0
    return _value(s, i, m, k)
end

_isdigit(b) = UInt8('0') <= b <= UInt8('9')

function _value(s, i, m::ToolMatcher, k::Int)
    ty = m.holes[k]
    if ty == :boolean
        for word in (b"true", b"false")
            c, j = _match_bytes(s, i, word)
            c == 1 && return 1
            c == 2 && return _seg(s, j, m, k + 1)
        end
        return 0
    elseif ty == :integer || ty == :number
        i > length(s) && return 1
        s[i] == UInt8('-') && (i += 1)
        ndig = 0
        while i <= length(s) && _isdigit(s[i])
            i += 1
            ndig += 1
        end
        i > length(s) && return 1
        if ty == :number && s[i] == UInt8('.')
            ndig > 0 || return 0
            i += 1
            nfrac = 0
            while i <= length(s) && _isdigit(s[i])
                i += 1
                nfrac += 1
            end
            i > length(s) && return 1
            return nfrac > 0 ? _seg(s, i, m, k + 1) : 0
        end
        return ndig > 0 ? _seg(s, i, m, k + 1) : 0
    else # :string — QUOTE strchar* QUOTE, where strchar is anything but "<|"
        c, i = _match_bytes(s, i, QUOTE_BYTES)
        c == 0 && return 0
        c == 1 && return 1
        while i <= length(s)
            if s[i] == UInt8('<')
                i == length(s) && return 1       # "<" could open either form
                if s[i + 1] == UInt8('|')        # must be the closing quote
                    c, j = _match_bytes(s, i, QUOTE_BYTES)
                    c == 0 && return 0
                    c == 1 && return 1
                    return _seg(s, j, m, k + 1)
                end
                i += 2
            else
                i += 1
            end
        end
        return 1
    end
end

"""
    call_state(matchers, bytes) -> :invalid | :prefix | :complete

Classify the call region generated so far (the bytes after `<|tool_call>`)
against the compiled tool matchers.
"""
function call_state(matchers, s::AbstractVector{UInt8})
    best = 0
    for m in matchers
        r = _seg(s, 1, m, 1)
        r == 2 && return :complete
        best = max(best, r)
    end
    return best == 1 ? :prefix : :invalid
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
