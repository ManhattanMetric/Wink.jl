# Quantized weights as ARRAY TYPES. Each type is an AbstractMatrix{Float32}
# whose storage is raw ggml blocks — parametric over the storage vector, so
# weights can be ZERO-COPY views straight into the GGUF mmap (13.4GB models
# never get copied). Because they satisfy the array interface, the generic
# forward passes run on them unchanged; dispatch selects the dequantizing
# mul! for the hot paths and exact per-element getindex for everything else.
#
#   q4_0 (type 2):  32 weights/block, 18 bytes: f16 scale + 16 nibble bytes;
#                   w = d * (q - 8). Fast fused mul!.
#   q8_0 (type 8):  32 weights/block, 34 bytes: f16 scale + 32 int8;
#                   w = d * q. Fast fused mul!.
#   q6_K (type 14): 256 weights/superblock, 210 bytes: 6-bit quants with
#                   per-16 int8 sub-scales and an f16 super-scale;
#                   w = d * sc * (q - 32). getindex only — it appears as the
#                   token-embedding table, which is gathered, not matmul'd.

module Quant

using LinearAlgebra
import KernelAbstractions
import KernelAbstractions: @kernel, @Const, @index
import Adapt

export Q4_0Matrix, Q8_0Matrix, Q6_KMatrix

# storage decides where the math runs: mmap views are CPU, device vectors GPU
_bytes_backend(d) =
    KernelAbstractions.get_backend(d isa SubArray ? parent(d) : d)

const QK = 32
const BLOCK = 18
const BLOCK8 = 34
const QK6 = 256
const BLOCK6 = 210

@inline _scale(data, off) = Float32(reinterpret(Float16,
    UInt16(data[off + 1]) | (UInt16(data[off + 2]) << 8)))

# ---- q4_0 ---------------------------------------------------------------------

struct Q4_0Matrix{D <: AbstractVector{UInt8}} <: AbstractMatrix{Float32}
    data::D
    nrow::Int
    ncol::Int
    function Q4_0Matrix(data::D, nrow::Integer, ncol::Integer) where {D}
        nrow % QK == 0 || error("q4_0 rows must be a multiple of $QK")
        length(data) == (nrow ÷ QK) * BLOCK * ncol ||
            error("q4_0 data size mismatch")
        new{D}(data, Int(nrow), Int(ncol))
    end
end

Base.size(A::Q4_0Matrix) = (A.nrow, A.ncol)

Base.@propagate_inbounds function Base.getindex(A::Q4_0Matrix, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    b, p = divrem(i - 1, QK)
    off = ((j - 1) * (A.nrow ÷ QK) + b) * BLOCK
    q = A.data[off + 2 + (p % 16) + 1]
    nib = p < 16 ? (q & 0x0f) : (q >> 4)
    return _scale(A.data, off) * (Float32(nib) - 8.0f0)
end

function LinearAlgebra.mul!(Y::StridedVecOrMat{Float32},
        At::Adjoint{Float32, <:Q4_0Matrix}, X::StridedVecOrMat{Float32})
    A = parent(At)
    size(X, 1) == A.nrow || throw(DimensionMismatch("q4_0 mul"))
    be = _bytes_backend(A.data)
    be isa KernelAbstractions.CPU || return _mul_ka!(_q4_mul_kernel!, be, Y, A, X,
        A.nrow ÷ QK)
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

# ---- q8_0 ---------------------------------------------------------------------

struct Q8_0Matrix{D <: AbstractVector{UInt8}} <: AbstractMatrix{Float32}
    data::D
    nrow::Int
    ncol::Int
    function Q8_0Matrix(data::D, nrow::Integer, ncol::Integer) where {D}
        nrow % QK == 0 || error("q8_0 rows must be a multiple of $QK")
        length(data) == (nrow ÷ QK) * BLOCK8 * ncol ||
            error("q8_0 data size mismatch")
        new{D}(data, Int(nrow), Int(ncol))
    end
end

Base.size(A::Q8_0Matrix) = (A.nrow, A.ncol)

Base.@propagate_inbounds function Base.getindex(A::Q8_0Matrix, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    b, p = divrem(i - 1, QK)
    off = ((j - 1) * (A.nrow ÷ QK) + b) * BLOCK8
    return _scale(A.data, off) * Float32(reinterpret(Int8, A.data[off + 2 + p + 1]))
end

function LinearAlgebra.mul!(Y::StridedVecOrMat{Float32},
        At::Adjoint{Float32, <:Q8_0Matrix}, X::StridedVecOrMat{Float32})
    A = parent(At)
    size(X, 1) == A.nrow || throw(DimensionMismatch("q8_0 mul"))
    be = _bytes_backend(A.data)
    be isa KernelAbstractions.CPU || return _mul_ka!(_q8_mul_kernel!, be, Y, A, X,
        A.nrow ÷ QK)
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

# ---- q6_K (element access only) -----------------------------------------------

struct Q6_KMatrix{D <: AbstractVector{UInt8}} <: AbstractMatrix{Float32}
    data::D
    nrow::Int
    ncol::Int
    function Q6_KMatrix(data::D, nrow::Integer, ncol::Integer) where {D}
        nrow % QK6 == 0 || error("q6_K rows must be a multiple of $QK6")
        length(data) == (nrow ÷ QK6) * BLOCK6 * ncol ||
            error("q6_K data size mismatch")
        new{D}(data, Int(nrow), Int(ncol))
    end
end

Base.size(A::Q6_KMatrix) = (A.nrow, A.ncol)

Base.@propagate_inbounds function Base.getindex(A::Q6_KMatrix, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    sb, r = divrem(i - 1, QK6)
    off = ((j - 1) * (A.nrow ÷ QK6) + sb) * BLOCK6
    half, n = divrem(r, 128)
    grp, l = divrem(n, 32)
    ql = off + half * 64
    qh = off + 128 + half * 32
    sc = off + 192 + half * 8
    data = A.data
    q = if grp == 0
        (data[ql + l + 1] & 0x0f) | ((data[qh + l + 1] & 0x03) << 4)
    elseif grp == 1
        (data[ql + 32 + l + 1] & 0x0f) | (((data[qh + l + 1] >> 2) & 0x03) << 4)
    elseif grp == 2
        (data[ql + l + 1] >> 4) | (((data[qh + l + 1] >> 4) & 0x03) << 4)
    else
        (data[ql + 32 + l + 1] >> 4) | (((data[qh + l + 1] >> 6) & 0x03) << 4)
    end
    is = sc + (l ÷ 16) + 2 * grp
    d = Float32(reinterpret(Float16,
        UInt16(data[off + 209]) | (UInt16(data[off + 210]) << 8)))
    return d * Float32(reinterpret(Int8, data[is + 1])) * (Float32(q) - 32.0f0)
end

function LinearAlgebra.mul!(Y::StridedVecOrMat{Float32},
        At::Adjoint{Float32, <:Q6_KMatrix}, X::StridedVecOrMat{Float32})
    A = parent(At)
    size(X, 1) == A.nrow || throw(DimensionMismatch("q6_K mul"))
    be = _bytes_backend(A.data)
    be isa KernelAbstractions.CPU || return _mul_ka!(_q6_mul_kernel!, be, Y, A, X,
        A.nrow ÷ QK6)
    nsb = A.nrow ÷ QK6
    data = A.data
    T = size(X, 2)
    Threads.@threads for j in 1:A.ncol
        colbase = (j - 1) * nsb * BLOCK6
        for t in 1:T
            acc = 0.0f0
            @inbounds for sb in 0:(nsb - 1)
                off = colbase + sb * BLOCK6
                d = Float32(reinterpret(Float16,
                    UInt16(data[off + 209]) | (UInt16(data[off + 210]) << 8)))
                s = 0.0f0
                for half in 0:1
                    ql = off + half * 64
                    qh = off + 128 + half * 32
                    sc = off + 192 + half * 8
                    xb = sb * QK6 + half * 128
                    for is in 0:1
                        s1 = s2 = s3 = s4 = 0.0f0
                        base = is * 16
                        @simd for l in base:(base + 15)
                            qlb = data[ql + l + 1]
                            qlb32 = data[ql + 32 + l + 1]
                            qhb = data[qh + l + 1]
                            q1 = Float32((qlb & 0x0f) | ((qhb & 0x03) << 4)) - 32.0f0
                            q2 = Float32((qlb32 & 0x0f) |
                                         (((qhb >> 2) & 0x03) << 4)) - 32.0f0
                            q3 = Float32((qlb >> 4) |
                                         (((qhb >> 4) & 0x03) << 4)) - 32.0f0
                            q4 = Float32((qlb32 >> 4) | ((qhb >> 6) << 4)) - 32.0f0
                            s1 = muladd(q1, X[xb + l + 1, t], s1)
                            s2 = muladd(q2, X[xb + 32 + l + 1, t], s2)
                            s3 = muladd(q3, X[xb + 64 + l + 1, t], s3)
                            s4 = muladd(q4, X[xb + 96 + l + 1, t], s4)
                        end
                        s = muladd(Float32(reinterpret(Int8, data[sc + is + 1])), s1, s)
                        s = muladd(Float32(reinterpret(Int8, data[sc + is + 3])), s2, s)
                        s = muladd(Float32(reinterpret(Int8, data[sc + is + 5])), s3, s)
                        s = muladd(Float32(reinterpret(Int8, data[sc + is + 7])), s4, s)
                    end
                end
                acc = muladd(d, s, acc)
            end
            @inbounds Y[j, t] = acc
        end
    end
    return Y
end

# ---- KernelAbstractions device paths ------------------------------------------
#
# One thread per output element (weight column × activation column); each
# dequantizes its blocks in registers. Written once, runs on every JuliaGPU
# backend (and the KA CPU backend, though the threaded loops above are faster
# there, which is why mul! branches on the storage's backend).

function _mul_ka!(kernel, be, Y, A, X, nblocks)
    kernel(be)(Y, A.data, X, Int32(nblocks); ndrange = (A.ncol, size(X, 2)))
    return Y
end

@kernel function _q4_mul_kernel!(Y, @Const(data), @Const(X), nb::Int32)
    j, t = @index(Global, NTuple)
    colbase = (j - 1) * Int(nb) * BLOCK
    acc = 0.0f0
    @inbounds for b in 0:(Int(nb) - 1)
        off = colbase + b * BLOCK
        s = 0.0f0
        xoff = b * QK
        for i in 1:16
            q = data[off + 2 + i]
            s = muladd(Float32(q & 0x0f) - 8.0f0, X[xoff + i, t], s)
            s = muladd(Float32(q >> 4) - 8.0f0, X[xoff + 16 + i, t], s)
        end
        acc = muladd(_scale(data, off), s, acc)
    end
    @inbounds Y[j, t] = acc
end

@kernel function _q8_mul_kernel!(Y, @Const(data), @Const(X), nb::Int32)
    j, t = @index(Global, NTuple)
    colbase = (j - 1) * Int(nb) * BLOCK8
    acc = 0.0f0
    @inbounds for b in 0:(Int(nb) - 1)
        off = colbase + b * BLOCK8
        s = 0.0f0
        xoff = b * QK
        for i in 1:QK
            s = muladd(Float32(reinterpret(Int8, data[off + 2 + i])),
                X[xoff + i, t], s)
        end
        acc = muladd(_scale(data, off), s, acc)
    end
    @inbounds Y[j, t] = acc
end

@kernel function _q6_mul_kernel!(Y, @Const(data), @Const(X), nsb::Int32)
    j, t = @index(Global, NTuple)
    colbase = (j - 1) * Int(nsb) * BLOCK6
    acc = 0.0f0
    @inbounds for sb in 0:(Int(nsb) - 1)
        off = colbase + sb * BLOCK6
        s = 0.0f0
        for half in 0:1
            ql = off + half * 64
            qh = off + 128 + half * 32
            sc = off + 192 + half * 8
            xb = sb * QK6 + half * 128
            for l in 0:31
                qlb = data[ql + l + 1]
                qlb32 = data[ql + 32 + l + 1]
                qhb = data[qh + l + 1]
                is = l >> 4
                q1 = Float32((qlb & 0x0f) | ((qhb & 0x03) << 4)) - 32.0f0
                q2 = Float32((qlb32 & 0x0f) |
                             (((qhb >> 2) & 0x03) << 4)) - 32.0f0
                q3 = Float32((qlb >> 4) | (((qhb >> 4) & 0x03) << 4)) - 32.0f0
                q4 = Float32((qlb32 >> 4) | ((qhb >> 6) << 4)) - 32.0f0
                s = muladd(Float32(reinterpret(Int8, data[sc + is + 1])) * q1,
                    X[xb + l + 1, t], s)
                s = muladd(Float32(reinterpret(Int8, data[sc + is + 3])) * q2,
                    X[xb + 32 + l + 1, t], s)
                s = muladd(Float32(reinterpret(Int8, data[sc + is + 5])) * q3,
                    X[xb + 64 + l + 1, t], s)
                s = muladd(Float32(reinterpret(Int8, data[sc + is + 7])) * q4,
                    X[xb + 96 + l + 1, t], s)
            end
        end
        acc = muladd(_scale(data, off + 208), s, acc)
    end
    @inbounds Y[j, t] = acc
end

# ---- shared interface ---------------------------------------------------------

const AnyQuant = Union{Q4_0Matrix, Q8_0Matrix, Q6_KMatrix}

# activations and caches allocated "like" quantized weights are plain arrays
Base.similar(A::AnyQuant, ::Type{T}, dims::Dims) where {T} = Array{T}(undef, dims)

KernelAbstractions.get_backend(A::AnyQuant) = _bytes_backend(A.data)

# device movement: MATERIALIZE the bytes before adapting — adapting an mmap
# view directly would rebuild the view over an adapted parent, i.e. upload
# the entire multi-GB file once per tensor
Adapt.adapt_structure(to, A::Q4_0Matrix) =
    Q4_0Matrix(Adapt.adapt(to, collect(A.data)), A.nrow, A.ncol)
Adapt.adapt_structure(to, A::Q8_0Matrix) =
    Q8_0Matrix(Adapt.adapt(to, collect(A.data)), A.nrow, A.ncol)
Adapt.adapt_structure(to, A::Q6_KMatrix) =
    Q6_KMatrix(Adapt.adapt(to, collect(A.data)), A.nrow, A.ncol)

end # module Quant
