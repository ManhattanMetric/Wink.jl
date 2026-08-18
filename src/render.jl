# Chat rendering as a generic function: one template family, one method.
#
# `render_chat(family, msgs; tools, add_assistant)` produces the exact prompt
# text a model family expects. Families are types, so supporting a new model
# is defining a method — not patching a matcher. Three methods here:
#
#   Gemma4Template  — hand-implemented from the model's own chat template
#                     ("Gemma 4 Canonical Chat Template", 2026-07-09;
#                     preserved as spike/gemma4_template.jinja on the
#                     olmoe-tuning branch), which llama.cpp's C matcher
#                     does not recognize. Validated byte-for-byte against
#                     Python jinja2 rendering the real template
#                     (spike/test_render.jl, olmoe-tuning branch; the
#                     shipped render coverage lives in test_local.jl).
#   NativeTemplate  — delegate to llama.cpp's C-side matcher for the families
#                     it does know (Nemotron et al.). Text-only: no tools.
#   ChatMLTemplate  — last-resort fallback, reported loudly by callers.
#
# Messages are NamedTuples or Dicts with `role` and `content`, plus optional
# `tool_calls` ([(id, name, args::Dict)-like]) on assistant messages and
# `tool_call_id` on role="tool" messages. BOS is never rendered as text —
# tokenize the result with add_special=true on the first decode instead.

abstract type ChatTemplate end

Base.@kwdef struct Gemma4Template <: ChatTemplate
    enable_thinking::Bool = false
end
struct Gemma3Template <: ChatTemplate end
struct NativeTemplate <: ChatTemplate
    template::String
end
struct ChatMLTemplate <: ChatTemplate end

# The pieces that END a model turn under each family — generation must stop
# on these. GGUF metadata often omits eot_token_id (gemma-4's does);
# llama.cpp compensates with a hardcoded piece list in its vocab loader,
# and this is our equivalent, keyed by family instead of guessed globally.
struct ZephyrTemplate <: ChatTemplate end

stop_pieces(::Gemma4Template) = ("<turn|>", "<eos>")
stop_pieces(::Gemma3Template) = ("<end_of_turn>", "<eos>")
stop_pieces(::ZephyrTemplate) = ("<|endoftext|>",)
stop_pieces(::ChatTemplate) = ("<|im_end|>", "<|endoftext|>", "<eos>", "<eot>")

# Field access over NamedTuples and Dicts alike.
field(m, k::Symbol, default = nothing) = m isa AbstractDict ?
    get(m, String(k), get(m, k, default)) :
    hasproperty(m, k) ? getproperty(m, k) : default
mrole(m) = String(something(field(m, :role), ""))
mcontent(m) = something(field(m, :content), "")

"""
    template_family(template_text; native_probe = nothing) -> ChatTemplate

Pick the rendering family for a model's GGUF chat template. Gemma-4's family
is recognized by its own markers; otherwise `native_probe` (a function that
returns `true` when llama.cpp's C matcher can render the template) selects
the native path, and ChatML is the loud last resort.
"""
function template_family(template_text::Union{Nothing, AbstractString};
        native_probe = nothing)
    template_text !== nothing && occursin("<|turn>", template_text) &&
        return Gemma4Template()
    template_text !== nothing && occursin("start_of_turn", template_text) &&
        return Gemma3Template()
    template_text !== nothing && occursin("<|user|>", template_text) &&
        occursin("<|assistant|>", template_text) &&
        return ZephyrTemplate()
    template_text !== nothing && native_probe !== nothing && native_probe() &&
        return NativeTemplate(String(template_text))
    return ChatMLTemplate()
end

# ---- ChatML -------------------------------------------------------------------

render_chat(::ChatMLTemplate, msgs; tools = (), add_assistant::Bool = true) =
    join("<|im_start|>$(mrole(m))\n$(mcontent(m))<|im_end|>\n" for m in msgs) *
    (add_assistant ? "<|im_start|>assistant\n" : "")

# ---- Zephyr (OLMo/OLMoE, Tulu-style) ------------------------------------------
#
# The <|role|> family, per llama.cpp's canonical zephyr rendering: BOS is
# never rendered as text (OLMoE sets add_bos_token = false anyway), each
# turn is "<|role|>\ncontent\n", and assistant turns close with the
# <|endoftext|> eos per the model's own template. Text-only — no canonical
# tool-call format; tools are withheld upstream, not degraded here.

function render_chat(::ZephyrTemplate, msgs; tools = (), add_assistant::Bool = true)
    isempty(tools) ||
        error("ZephyrTemplate is text-only (OLMo/zephyr has no canonical " *
              "tool-call format); tools require another family")
    io = IOBuffer()
    for m in msgs
        role = mrole(m)
        content = strip(mcontent(m))
        if role == "assistant"
            print(io, "<|assistant|>\n", content, "<|endoftext|>\n")
        else
            print(io, "<|", role in ("system", "developer") ? "system" : "user",
                "|>\n", content, "\n")
        end
    end
    add_assistant && print(io, "<|assistant|>\n")
    return String(take!(io))
end

# ---- Gemma 3 ------------------------------------------------------------------
#
# The <start_of_turn> family. No system role: a leading system message is
# folded into the first user turn, per Google's own convention. Text-only —
# gemma-3 has no canonical tool-call format, so tools are refused rather
# than degraded.

function render_chat(::Gemma3Template, msgs; tools = (), add_assistant::Bool = true)
    isempty(tools) ||
        error("Gemma3Template is text-only (gemma-3 has no canonical " *
              "tool-call format); tools require another family")
    io = IOBuffer()
    sys = ""
    for m in msgs
        role = mrole(m)
        content = strip(mcontent(m))
        if role in ("system", "developer")
            sys = isempty(sys) ? String(content) : sys * "\n\n" * content
            continue
        end
        if role == "user" && !isempty(sys)
            content = sys * "\n\n" * content
            sys = ""
        end
        print(io, "<start_of_turn>", role == "assistant" ? "model" : "user",
            "\n", content, "<end_of_turn>\n")
    end
    isempty(sys) ||   # system with no user turn yet
        print(io, "<start_of_turn>user\n", sys, "<end_of_turn>\n")
    add_assistant && print(io, "<start_of_turn>model\n")
    return String(take!(io))
end

# ---- Gemma 4 ------------------------------------------------------------------
#
# Faithful transcription of gemma4_template.jinja. Structure: an optional
# system turn carrying `<|think|>`, the system prompt, and `<|tool>`
# declarations; then turns as `<|turn>role\n … <turn|>\n` with assistant
# rendered as "model". Tool calls are `<|tool_call>call:name{k:v}<tool_call|>`
# followed by their responses forward-scanned from role="tool" messages —
# and a model turn that ends on tool responses is left OPEN, so the model
# resumes the same turn after results: the agentic-loop shape. When thinking
# is disabled, the generation prompt pre-closes an empty thought channel.

# jinja's format_argument: strings get the dedicated quote token; only
# mapping KEYS obey escape_keys.
function g4_argument(io::IO, v; escape_keys::Bool = true)
    if v === nothing
        print(io, "null")
    elseif v isa AbstractString
        print(io, "<|\"|>", v, "<|\"|>")
    elseif v isa Bool
        print(io, v ? "true" : "false")
    elseif v isa AbstractDict
        print(io, "{")
        for (i, k) in enumerate(g4_sortkeys(v))
            i > 1 && print(io, ",")
            escape_keys ? print(io, "<|\"|>", k, "<|\"|>") : print(io, k)
            print(io, ":")
            g4_argument(io, v[k]; escape_keys)
        end
        print(io, "}")
    elseif v isa Union{AbstractVector, Tuple}
        print(io, "[")
        for (i, x) in enumerate(v)
            i > 1 && print(io, ",")
            g4_argument(io, x; escape_keys)
        end
        print(io, "]")
    else
        print(io, v)
    end
end

# jinja dictsort: by key, case-insensitive.
g4_sortkeys(d::AbstractDict) = sort(collect(keys(d)); by = k -> lowercase(String(k)))

# jinja format_parameters: one property as `key:{…,type:<|"|>T<|"|>}`.
function g4_parameters(io::IO, properties::AbstractDict, required)
    first_prop = true
    for key in g4_sortkeys(properties)
        v = properties[key]
        first_prop || print(io, ",")
        first_prop = false
        print(io, key, ":{")
        comma = false
        desc = field(v, :description)
        if desc !== nothing && desc != ""
            print(io, "description:<|\"|>", desc, "<|\"|>")
            comma = true
        end
        typ = uppercase(String(something(field(v, :type), "")))
        if typ == "STRING" && field(v, :enum) !== nothing
            comma && print(io, ",")
            comma = true
            print(io, "enum:")
            g4_argument(io, field(v, :enum))
        elseif typ == "ARRAY" && field(v, :items) isa AbstractDict &&
               !isempty(field(v, :items))
            comma && print(io, ",")
            comma = true
            items = field(v, :items)
            print(io, "items:{")
            first_item = true
            for ik in g4_sortkeys(items)
                iv = items[ik]
                iv === nothing && continue
                first_item || print(io, ",")
                first_item = false
                if String(ik) == "properties"
                    print(io, "properties:{")
                    iv isa AbstractDict &&
                        g4_parameters(io, iv, something(field(items, :required), []))
                    print(io, "}")
                elseif String(ik) == "required"
                    print(io, "required:[")
                    for (j, r) in enumerate(iv)
                        j > 1 && print(io, ",")
                        print(io, "<|\"|>", r, "<|\"|>")
                    end
                    print(io, "]")
                elseif String(ik) == "type"
                    print(io, "type:")
                    g4_argument(io, iv isa AbstractString ? uppercase(iv) :
                                    [uppercase(String(x)) for x in iv])
                else
                    print(io, ik, ":")
                    g4_argument(io, iv)
                end
            end
            print(io, "}")
        end
        if field(v, :nullable) == true
            comma && print(io, ",")
            comma = true
            print(io, "nullable:true")
        end
        if typ == "OBJECT"
            props = field(v, :properties)
            if props isa AbstractDict
                comma && print(io, ",")
                comma = true
                print(io, "properties:{")
                g4_parameters(io, props, something(field(v, :required), []))
                print(io, "}")
            end
            req = field(v, :required)
            if req !== nothing && !isempty(req)
                comma && print(io, ",")
                comma = true
                print(io, "required:[")
                for (j, r) in enumerate(req)
                    j > 1 && print(io, ",")
                    print(io, "<|\"|>", r, "<|\"|>")
                end
                print(io, "]")
            end
        end
        comma && print(io, ",")
        print(io, "type:<|\"|>", typ, "<|\"|>}")
    end
end

# jinja format_function_declaration.
function g4_declaration(tool)
    fn = something(field(tool, :function), tool)
    io = IOBuffer()
    print(io, "declaration:", field(fn, :name),
        "{description:<|\"|>", field(fn, :description), "<|\"|>")
    params = field(fn, :parameters)
    if params !== nothing && !isempty(params)
        print(io, ",parameters:{")
        props = field(params, :properties)
        if props !== nothing && !isempty(props)
            print(io, "properties:{")
            g4_parameters(io, props, something(field(params, :required), []))
            print(io, "},")
        end
        req = field(params, :required)
        if req !== nothing && !isempty(req)
            print(io, "required:[")
            for (j, r) in enumerate(req)
                j > 1 && print(io, ",")
                print(io, "<|\"|>", r, "<|\"|>")
            end
            print(io, "],")
        end
        typ = field(params, :type)
        typ !== nothing && print(io, "type:<|\"|>", uppercase(String(typ)), "<|\"|>}")
    end
    print(io, "}")
    return String(take!(io))
end

# jinja strip_thinking: drop `<|channel> … <channel|>` spans from model text.
function g4_strip_thinking(text::AbstractString)
    out = IOBuffer()
    for part in split(text, "<channel|>")
        print(out, occursin("<|channel>", part) ?
                   first(split(part, "<|channel>"; limit = 2)) : part)
    end
    return strip(String(take!(out)))
end

function g4_tool_response_block(io::IO, name, response)
    print(io, "<|tool_response>")
    if response isa AbstractDict
        print(io, "response:", name, "{")
        for (i, k) in enumerate(g4_sortkeys(response))
            i > 1 && print(io, ",")
            print(io, k, ":")
            g4_argument(io, response[k]; escape_keys = false)
        end
        print(io, "}")
    else
        print(io, "response:", name, "{value:")
        g4_argument(io, response; escape_keys = false)
        print(io, "}")
    end
    print(io, "<tool_response|>")
end

function render_chat(t::Gemma4Template, msgs; tools = (), add_assistant::Bool = true)
    io = IOBuffer()
    thinking = t.enable_thinking
    first_system = !isempty(msgs) && mrole(msgs[1]) in ("system", "developer")
    body = first_system ? msgs[2:end] : msgs

    last_type = nothing   # ns.prev_message_type
    if thinking || !isempty(tools) || first_system
        print(io, "<|turn>system\n")
        if thinking
            print(io, "<|think|>\n")
            last_type = :think
        end
        first_system && print(io, strip(mcontent(msgs[1])))
        for tool in tools
            print(io, "<|tool>", strip(g4_declaration(tool)), "<tool|>")
        end
        isempty(tools) || (last_type = :tool)
        print(io, "<turn|>\n")
    end

    last_user_idx = something(findlast(m -> mrole(m) == "user", collect(body)), 0)
    prev_nontool = nothing
    n = length(body)
    for i in 1:n
        m = body[i]
        mrole(m) == "tool" && continue        # consumed by the forward scan
        last_type = nothing
        role = mrole(m) == "assistant" ? "model" : mrole(m)
        continued = role == "model" && prev_nontool == "assistant"
        continued || print(io, "<|turn>", role, "\n")

        reasoning = something(field(m, :reasoning), field(m, :reasoning_content, ""))
        if !isempty(reasoning) && i > last_user_idx
            print(io, "<|channel>thought\n", reasoning, "\n<channel|>")
        end

        tcs = something(field(m, :tool_calls), ())
        responses_emitted = false
        if !isempty(tcs)
            for tc in tcs
                fn = something(field(tc, :function), tc)
                print(io, "<|tool_call>call:", field(fn, :name), "{")
                args = something(field(fn, :arguments), field(fn, :args, nothing))
                args isa AbstractString &&
                    error("tool_calls arguments must be a Dict, not a JSON string")
                if args isa AbstractDict
                    for (j, k) in enumerate(g4_sortkeys(args))
                        j > 1 && print(io, ",")
                        print(io, k, ":")
                        g4_argument(io, args[k]; escape_keys = false)
                    end
                end
                print(io, "}<tool_call|>")
            end
            last_type = :tool_call
            for j in (i + 1):n
                mrole(body[j]) == "tool" || break
                follow = body[j]
                name = something(field(follow, :name), "unknown")
                for tc in tcs
                    if field(tc, :id) !== nothing &&
                       field(tc, :id) == field(follow, :tool_call_id)
                        name = field(something(field(tc, :function), tc), :name)
                    end
                end
                g4_tool_response_block(io, name, mcontent(follow))
                responses_emitted = true
                last_type = :tool_response
            end
        end

        content = mcontent(m)
        c = role == "model" ? g4_strip_thinking(content) : strip(content)
        print(io, c)
        has_content = !isempty(strip(c))

        next_role = nothing
        for j in (i + 1):n
            if mrole(body[j]) != "tool"
                next_role = mrole(body[j])
                break
            end
        end
        continues_next = role == "model" && next_role == "assistant" &&
                         (isempty(tcs) || responses_emitted)

        if last_type === :tool_call && !responses_emitted
            print(io, "<|tool_response>")
        elseif continues_next
        elseif !(responses_emitted && !has_content && next_role === nothing)
            print(io, "<turn|>\n")
        end
        prev_nontool = mrole(m)
    end

    if add_assistant
        if last_type !== :tool_call && last_type !== :tool_response
            print(io, "<|turn>model\n")
            thinking || print(io, "<|channel>thought\n<channel|>")
        elseif last_type === :tool_response && thinking
            print(io, "<|channel>thought\n")
        end
    end
    return String(take!(io))
end

# ---- llama.cpp native ---------------------------------------------------------

function render_chat(t::NativeTemplate, msgs; tools = (), add_assistant::Bool = true)
    isempty(tools) ||
        error("NativeTemplate cannot render tools; this family needs its own method")
    roles = String[mrole(m) for m in msgs]
    contents = String[String(mcontent(m)) for m in msgs]
    cmsgs = [L.llama_chat_message(Ptr{Cchar}(pointer(roles[i])),
                 Ptr{Cchar}(pointer(contents[i]))) for i in eachindex(roles)]
    buf = Vector{UInt8}(undef, 16384)
    GC.@preserve roles contents begin
        n = L.llama_chat_apply_template(t.template, cmsgs, length(cmsgs),
            add_assistant, buf, length(buf))
        n < 0 && error("llama.cpp's matcher failed on a template it probed as supported")
        if n > length(buf)
            resize!(buf, n)
            n = L.llama_chat_apply_template(t.template, cmsgs, length(cmsgs),
                add_assistant, buf, length(buf))
            n < 0 && error("llama.cpp's matcher failed on re-render")
        end
        return String(buf[1:n])
    end
end
