function nemotron_rmsnorm(x::AbstractArray{T,3}, weight::AbstractVector{T}, eps::Real) where {T}
    # x: (B, T, D), weight: (D,)
    B, TT, D = size(x)
    out = similar(x)
    w32 = Float32.(weight)
    @inbounds for b in 1:B, t in 1:TT
        var = zero(Float32)
        for d in 1:D
            xd = Float32(x[b, t, d])
            var += xd * xd
        end
        var /= D
        inv = 1f0 / sqrt(var + Float32(eps))
        for d in 1:D
            out[b, t, d] = T(Float32(x[b, t, d]) * inv * w32[d])
        end
    end
    out
end

"""
Grouped RMS norm on `x` then multiply by `silu(gate)` (Mamba2 gated norm, `norm_before_gate=false` style).
`n_groups` segments of size `D ÷ n_groups`.
"""
function mamba_rmsnorm_gated(
    x::AbstractArray{T,3},
    weight::AbstractVector{T},
    gate::AbstractArray{T,3},
    eps::Real,
    n_groups::Int,
) where {T}
    B, TT, D = size(x)
    @assert D % n_groups == 0
    gs = D ÷ n_groups
    outf = zeros(Float32, B, TT, D)
    wf = Float32.(weight)
    xf = Float32.(x)
    gf = Float32.(gate)
    @inbounds for b in 1:B, t in 1:TT, g in 1:n_groups
        off = (g - 1) * gs
        var = zero(Float32)
        for i in 1:gs
            v = xf[b, t, off+i]
            var += v * v
        end
        var /= gs
        inv = 1f0 / sqrt(var + Float32(eps))
        for i in 1:gs
            idx = off + i
            outf[b, t, idx] = xf[b, t, idx] * inv * wf[idx]
        end
    end
    silu_g = gf .* (1f0 ./ (1f0 .+ exp.(-gf)))
    y = outf .* silu_g
    return T.(y)
end
