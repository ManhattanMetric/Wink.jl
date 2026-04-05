module Wink

using Functors

include("nemotron_h/config.jl")
include("nemotron_h/utils.jl")
include("nemotron_h/norms.jl")
include("nemotron_h/mamba2.jl")
include("nemotron_h/attention.jl")
include("nemotron_h/mlp.jl")
include("nemotron_h/block.jl")
include("nemotron_h/model.jl")
include("nemotron_h/gguf.jl")

export NemotronHConfig,
    nemotron_3_nano_4b_bf16,
    layers_block_type,
    NemotronHModel,
    NemotronHCausalLM,
    load_nemotron_h_gguf!,
    gguf_dequant_dict

end
