# Quantized weights as an ARRAY TYPE. Q4_0Matrix is an AbstractMatrix{Float32}
# whose storage is ggml q4_0 blocks (per 32 weights: one f16 scale + 16 bytes
# of 4-bit quants; value = scale * (q - 8)). Because it satisfies the array
# interface, the existing generic forward passes run on it UNCHANGED — the
# only thing that changes is which `mul!` method dispatch selects. The fast
# path dequantizes inside the dot-product loop, so full-precision weights
# never materialize: memory traffic is the quantized bytes, which is the
# entire point for models whose F16 form doesn't fit the machine.
#
# CPU implementation here (threaded, SIMD inner loop); the KernelAbstractions
# port of the same loop is the follow-up that takes this to GPUs.

module Quant

using LinearAlgebra
import KernelAbstractions

export Q4_0Matrix, Q8_0Matrix

const QK = 32          # weights per block
const BLOCK = 18       # q4_0 bytes per block: 2 (f16 scale) + 16 (nibbles)
const BLOCK8 = 34      # q8_0 bytes per block: 2 (f16 scale) + 32 (int8)

"""
    Q4_0Matrix(data, nrow, ncol)

ggml q4_0 tensor as a matrix: `nrow` is the contiguous (input) dimension,
blocks run down each column. Indexable like any matrix (slow, exact);
multiplied via the specialized `mul!` (fast, exact — no activation
requantization, unlike llama.cpp's integer path).
"""
struct Q4_0Matrix <: AbstractMatrix{Float32}
    data::Vector{UInt8}
    nrow::Int
    ncol::Int
    function Q4_0Matrix(data::Vector{UInt8}, nrow::Integer, ncol::Integer)
        nrow % QK == 0 || error("q4_0 rows must be a multiple of $QK")
        length(data) == (nrow ÷ QK) * BLOCK * ncol ||
            error("q4_0 data size mismatch")
        new(data, Int(nrow), Int(ncol))
    end
end

Base.size(A::Q4_0Matrix) = (A.nrow, A.ncol)

@inline _scale(data, off) = Float32(reinterpret(Float16,
    UInt16(data[off + 1]) | (UInt16(data[off + 2]) << 8)))

Base.@propagate_inbounds function Base.getindex(A::Q4_0Matrix, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    b, p = divrem(i - 1, QK)
    off = ((j - 1) * (A.nrow ÷ QK) + b) * BLOCK
    q = A.data[off + 2 + (p % 16) + 1]
    nib = p < 16 ? (q & 0x0f) : (q >> 4)
    return _scale(A.data, off) * (Float32(nib) - 8.0f0)
end

# similar() on a quantized matrix yields a plain array of the requested shape
# (KV caches, activations); you cannot allocate "empty quantized weights".
Base.similar(A::Q4_0Matrix, ::Type{T}, dims::Dims) where {T} =
    Array{T}(undef, dims)

# The hot path: Y = Aᵀ X with dequantization fused into the dot product.
function LinearAlgebra.mul!(Y::StridedVecOrMat{Float32},
        At::Adjoint{Float32, Q4_0Matrix}, X::StridedVecOrMat{Float32})
    A = parent(At)
    size(X, 1) == A.nrow || throw(DimensionMismatch("q4_0 mul"))
    nb = A.nrow ÷ QK
    data = A.data
    T = size(X, 2)
    Threads.@threads for j in 1:A.ncol
        colbase = (j - 1) * nb * BLOCK
        for t in 1:T
            acc = 0.0f0
            @inbounds for b in 0:(nb - 1)
                off = colbase + b * BLOCK
                s = 0.0f0
                xoff = b * QK
                @simd for i in 1:16
                    q = data[off + 2 + i]
                    s = muladd(Float32(q & 0x0f) - 8.0f0, X[xoff + i, t], s)
                    s = muladd(Float32(q >> 4) - 8.0f0, X[xoff + 16 + i, t], s)
                end
                acc = muladd(_scale(data, off), s, acc)
            end
            @inbounds Y[j, t] = acc
        end
    end
    return Y
end

"""
    Q8_0Matrix(data, nrow, ncol)

ggml q8_0: per 32 weights, one f16 scale and 32 signed bytes; value = d * q.
Same array-interface contract as Q4_0Matrix.
"""
struct Q8_0Matrix <: AbstractMatrix{Float32}
    data::Vector{UInt8}
    nrow::Int
    ncol::Int
    function Q8_0Matrix(data::Vector{UInt8}, nrow::Integer, ncol::Integer)
        nrow % QK == 0 || error("q8_0 rows must be a multiple of $QK")
        length(data) == (nrow ÷ QK) * BLOCK8 * ncol ||
            error("q8_0 data size mismatch")
        new(data, Int(nrow), Int(ncol))
    end
end

Base.size(A::Q8_0Matrix) = (A.nrow, A.ncol)

Base.@propagate_inbounds function Base.getindex(A::Q8_0Matrix, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    b, p = divrem(i - 1, QK)
    off = ((j - 1) * (A.nrow ÷ QK) + b) * BLOCK8
    return _scale(A.data, off) * Float32(reinterpret(Int8, A.data[off + 2 + p + 1]))
end

Base.similar(A::Q8_0Matrix, ::Type{T}, dims::Dims) where {T} =
    Array{T}(undef, dims)

function LinearAlgebra.mul!(Y::StridedVecOrMat{Float32},
        At::Adjoint{Float32, Q8_0Matrix}, X::StridedVecOrMat{Float32})
    A = parent(At)
    size(X, 1) == A.nrow || throw(DimensionMismatch("q8_0 mul"))
    nb = A.nrow ÷ QK
    data = A.data
    T = size(X, 2)
    Threads.@threads for j in 1:A.ncol
        colbase = (j - 1) * nb * BLOCK8
        for t in 1:T
            acc = 0.0f0
            @inbounds for b in 0:(nb - 1)
                off = colbase + b * BLOCK8
                s = 0.0f0
                xoff = b * QK
                @simd for i in 1:QK
                    s = muladd(Float32(reinterpret(Int8, data[off + 2 + i])),
                        X[xoff + i, t], s)
                end
                acc = muladd(_scale(data, off), s, acc)
            end
            @inbounds Y[j, t] = acc
        end
    end
    return Y
end

# quantized weights live in host memory; the GPU story for them is the
# KernelAbstractions dequant-matmul kernel, not device residency of this type
KernelAbstractions.get_backend(::Q4_0Matrix) = KernelAbstractions.CPU()
KernelAbstractions.get_backend(::Q8_0Matrix) = KernelAbstractions.CPU()

end # module Quant
