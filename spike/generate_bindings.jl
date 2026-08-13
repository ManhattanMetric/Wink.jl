# Generate LibLlama.jl from the vendored llama.cpp headers with Clang.jl.
# Rerun after bumping the vendored release (and re-fetching vendor/include).
#
# Run: julia --project=spike spike/generate_bindings.jl

using Clang.Generators

const VENDOR = joinpath(@__DIR__, "vendor")
const INCLUDE = joinpath(VENDOR, "include")

options = Dict{String, Any}(
    "general" => Dict{String, Any}(
        "module_name" => "LibLlama",
        "library_name" => "libllama",
        "output_file_path" => joinpath(@__DIR__, "LibLlama.jl"),
        "prologue_file_path" => joinpath(@__DIR__, "prologue.jl"),
        # ggml.h function-like macros are C-source conveniences that translate
        # to invalid Julia; they are irrelevant to the binding surface.
        "output_ignorelist" => [
            "GGML_TENSOR_[A-Z0-9_]*LOCALS[0-9_]*",
            "GGML_NORETURN", "GGML_UNUSED[A-Z_]*", "GGML_RESTRICT",
        ],
    ),
    "codegen" => Dict{String, Any}(
        "use_julia_bool" => true,
        # NOT always_NUL_terminated_string: llama.h uses `char *` both for
        # input strings and for output buffers (llama_token_to_piece); Cstring
        # conversion would reject the buffer case. Ptr{Cchar} handles both.
    ),
)

args = get_default_args()
push!(args, "-I$INCLUDE")

ctx = create_context([joinpath(INCLUDE, "llama.h")], args, options)
build!(ctx)

println("wrote ", options["general"]["output_file_path"])
