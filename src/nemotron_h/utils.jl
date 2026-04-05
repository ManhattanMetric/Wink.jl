silu(x) = x ./ (1 .+ exp.(-x))
softplus(x) = max.(x, zero(x)) .+ log.(1 .+ exp.(.-abs.(x)))
relu2(x) = (max.(x, zero(x))) .^ 2

function apply_mask_to_padding_states(hidden_states::AbstractArray{T,3}, attention_mask) where {T}
    attention_mask === nothing && return hidden_states
    m = reshape(attention_mask, size(attention_mask, 1), size(attention_mask, 2), 1)
    return hidden_states .* m
end

function pad_tensor_by_size(input_tensor::AbstractArray{Tx,3}, pad_size::Int) where {Tx}
    pad_size == 0 && return input_tensor
    B, seq, R = size(input_tensor)
    z = zero(Tx)
    out = similar(input_tensor, B, seq + pad_size, R)
    out[:, 1:seq, :] .= input_tensor
    out[:, seq+1:end, :] .= z
    out
end

function pad_tensor_by_size(input_tensor::AbstractArray{Tx,4}, pad_size::Int) where {Tx}
    pad_size == 0 && return input_tensor
    B, seq, H, D = size(input_tensor)
    z = zero(Tx)
    out = similar(input_tensor, B, seq + pad_size, H, D)
    out[:, 1:seq, :, :] .= input_tensor
    out[:, seq+1:end, :, :] .= z
    out
end

function reshape_into_chunks(input_tensor::AbstractArray, pad_size::Int, chunk_size::Int)
    x = pad_tensor_by_size(input_tensor, pad_size)
    B, Tp, rest... = size(x)
    @assert Tp % chunk_size == 0
    nchunks = Tp ÷ chunk_size
    if length(rest) == 1
        return reshape(x, B, nchunks, chunk_size, rest[1])
    elseif length(rest) == 2
        return reshape(x, B, nchunks, chunk_size, rest[1], rest[2])
    else
        error("unsupported rank")
    end
end

"""`segment_sum` for any rank ≥1; last dimension is the chunk axis (matches HF)."""
function segment_sum_lastdim(input_tensor::AbstractArray{T}) where {T}
    chunk_size = size(input_tensor)[end]
    lead = size(input_tensor)[1:end-1]
    M = isempty(lead) ? 1 : prod(lead)
    flat = reshape(input_tensor, M, chunk_size)
    neginf = T(-Inf)
    expanded = similar(flat, M, chunk_size, chunk_size)
    @inbounds for m in 1:M, a in 1:chunk_size, b in 1:chunk_size
        expanded[m, a, b] = b < a ? flat[m, b] : zero(T)
    end
    tensor_segsum = cumsum(expanded; dims=2)
    @inbounds for m in 1:M, a in 1:chunk_size, b in 1:chunk_size
        if b > a
            tensor_segsum[m, a, b] = neginf
        end
    end
    reshape(tensor_segsum, lead..., chunk_size, chunk_size)
end

const segment_sum = segment_sum_lastdim

"""Additive causal mask (B=1): allowed pairs stay 0, blocked pairs get -Inf."""
function causal_additive_mask(seqlen::Int, ::Type{F} = Float32) where {F<:AbstractFloat}
    minv = typemin(F)
    m = zeros(F, 1, 1, seqlen, seqlen)
    for i in 1:seqlen, j in 1:seqlen
        if j > i
            m[1, 1, i, j] = minv
        end
    end
    m
end

function softmax4(x::AbstractArray{Float32,4}; dims::Int = 4)
    mx = maximum(x; dims=dims)
    ex = exp.(x .- mx)
    ex ./ sum(ex; dims=dims)
end

function sdpa_causal(
    q::AbstractArray{Float32,4},
    k::AbstractArray{Float32,4},
    v::AbstractArray{Float32,4},
    scale::Float32,
    additive_mask = nothing,
)
    B, H, T, Dh = size(q)
    @assert size(k) == (B, H, T, Dh) && size(v) == (B, H, T, Dh)
    out = zeros(Float32, B, H, T, Dh)
    mask = causal_additive_mask(T, Float32)
    for b in 1:B, h in 1:H, i in 1:T
        logits = zeros(Float32, T)
        for j in 1:T
            s = zero(Float32)
            for d in 1:Dh
                s += q[b, h, i, d] * k[b, h, j, d]
            end
            extra = zero(Float32)
            if additive_mask !== nothing
                extra = additive_mask[b, 1, i, j]
            end
            logits[j] = s * scale + mask[1, 1, i, j] + extra
        end
        logits .= logits .- maximum(logits)
        w = exp.(logits)
        w ./= sum(w)
        for d in 1:Dh
            acc = zero(Float32)
            for j in 1:T
                acc += w[j] * v[b, h, j, d]
            end
            out[b, h, i, d] = acc
        end
    end
    out
end
