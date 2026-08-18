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
import SIMD
import SIMD: Vec, vload

export Q4_0Matrix, Q4_1Matrix, Q8_0Matrix, Q6_KMatrix, Q4_0Stack

# storage decides where the math runs: mmap views are CPU, device vectors GPU
_bytes_backend(d) =
    KernelAbstractions.get_backend(d isa SubArray ? parent(d) : d)

# ---- persistent worker pool ---------------------------------------------------
#
# Threads.@threads pays task-spawn + join-barrier per call — ~350 matmuls
# per generated token made that ~20% of thread time in profiles. Here
# workers park on auto-reset Events (zero CPU while Wink idles at the
# REPL), wake once per job, and self-balance by atomically stealing column
# chunks. The caller participates and yields until the last worker drains.

const POOL_LOCK = ReentrantLock()
const POOL_EVENTS = Base.Event[]
const JOB_FN = Ref{Any}(nothing)
const JOB_N = Ref(0)
const JOB_NEXT = Threads.Atomic{Int}(1)
const JOB_PENDING = Threads.Atomic{Int}(0)
const POOL_CHUNK = 32

function _pool_worker(ev::Base.Event)
    while true
        wait(ev)
        f = JOB_FN[]
        n = JOB_N[]
        while true
            i = Threads.atomic_add!(JOB_NEXT, POOL_CHUNK)
            i > n && break
            f(i, min(i + POOL_CHUNK - 1, n))
        end
        Threads.atomic_sub!(JOB_PENDING, 1)
    end
end

function _pool_init!()
    isempty(POOL_EVENTS) || return
    for _ in 1:max(Threads.nthreads() - 1, 0)
        ev = Base.Event(true)
        push!(POOL_EVENTS, ev)
        Threads.@spawn _pool_worker(ev)
    end
end

# run f(lo, hi) over disjoint chunks of 1:n across the pool + caller
function parallel_cols(f, n::Int)
    lock(POOL_LOCK) do
        _pool_init!()
        if isempty(POOL_EVENTS) || n < 2 * POOL_CHUNK
            f(1, n)
            return
        end
        JOB_FN[] = f
        JOB_N[] = n
        JOB_NEXT[] = 1
        JOB_PENDING[] = length(POOL_EVENTS)
        foreach(notify, POOL_EVENTS)
        while true
            i = Threads.atomic_add!(JOB_NEXT, POOL_CHUNK)
            i > n && break
            f(i, min(i + POOL_CHUNK - 1, n))
        end
        while JOB_PENDING[] > 0
            yield()
        end
        JOB_FN[] = nothing
    end
end

# ---- int8 dot-product path (aarch64) ------------------------------------------
#
# llama.cpp's core CPU trick, in Julia: quantize ACTIVATIONS to q8_0 blocks
# once per matmul, then the q4_0 kernel is integer SIMD — ARM's SDOT
# instruction does 16 int8 multiply-accumulates per lane-group, where the
# f32 path pays nibble-unpack + convert + 4-lane muladds. Weights stay in
# their mmap blocks; only the block scales meet Float32. On non-aarch64
# the exact-f32 kernels below remain the (portable) path.

const USE_SDOT = Sys.ARCH === :aarch64

@inline _sdot(acc::SIMD.LVec{4, Int32}, a::SIMD.LVec{16, Int8},
    b::SIMD.LVec{16, Int8}) =
    ccall("llvm.aarch64.neon.sdot.v4i32.v16i8", llvmcall, SIMD.LVec{4, Int32},
        (SIMD.LVec{4, Int32}, SIMD.LVec{16, Int8}, SIMD.LVec{16, Int8}),
        acc, a, b)

@inline sdot(acc::Vec{4, Int32}, a::Vec{16, Int8}, b::Vec{16, Int8}) =
    Vec(_sdot(acc.data, a.data, b.data))

# activations as q8_0: per 32-block f32 scale + 32 int8 (llama.cpp parity:
# d = amax/127, round-to-nearest)
struct Q8Act
    d::Matrix{Float32}    # nb × T
    q::Matrix{Int8}       # nrow × T
end

Q8Act(nrow::Int, T::Int) = Q8Act(Matrix{Float32}(undef, nrow ÷ QK, T),
    Matrix{Int8}(undef, nrow, T))

# task-local reuse: generation quantizes ~350 activation columns per token
# through here; the buffers are identical run to run. (Measured note: the
# broader 22MB/token of forward-pass temporaries costs only ~1.4% in GC —
# Julia's generational collector makes a full preallocated-plan rewrite of
# the oracle-validated step! a bad risk/reward trade.)
function _q8_scratch(nrow::Int, T::Int)
    d = get!(() -> Dict{Tuple{Int, Int}, Q8Act}(), task_local_storage(),
        :wink_q8act)::Dict{Tuple{Int, Int}, Q8Act}
    return get!(() -> Q8Act(nrow, T), d, (nrow, T))
end

function quantize_q8!(a::Q8Act, X)
    nb = size(X, 1) ÷ QK
    @inbounds for t in 1:size(X, 2), b in 0:(nb - 1)
        amax = 0.0f0
        base = b * QK
        for i in 1:QK
            amax = max(amax, abs(X[base + i, t]))
        end
        d = amax / 127.0f0
        id = d == 0.0f0 ? 0.0f0 : 1.0f0 / d
        a.d[b + 1, t] = d
        for i in 1:QK
            a.q[base + i, t] = round(Int8, X[base + i, t] * id)
        end
    end
    return a
end

# one q4_0 column (nb blocks at byte offset off0) · one quantized activation
# column: two SDOTs per block (low/high nibbles), horizontal add, scale by
# d_w * d_a
@inline function _q4_sdot_col(wptr::Ptr{UInt8}, off0::Int, nb::Int,
        qptr::Ptr{Int8}, dact::AbstractVector{Float32})
    tot = 0.0f0
    eight = Vec{16, Int8}(Int8(8))
    @inbounds for b in 0:(nb - 1)
        off = off0 + b * BLOCK
        v = vload(Vec{16, UInt8}, wptr + off + 2)
        lo = reinterpret(Vec{16, Int8}, v & 0x0f) - eight
        hi = reinterpret(Vec{16, Int8}, v >> 4) - eight
        ap = qptr + b * QK
        acc = sdot(sdot(Vec{4, Int32}(0), lo, vload(Vec{16, Int8}, ap)),
            hi, vload(Vec{16, Int8}, ap + 16))
        dw = Float32(reinterpret(Float16,
            UInt16(unsafe_load(wptr + off)) |
            (UInt16(unsafe_load(wptr + off + 1)) << 8)))
        tot = muladd(Float32(sum(acc)), dw * dact[b + 1], tot)
    end
    return tot
end

const QK = 32
const BLOCK = 18
const BLOCK8 = 34
const BLOCK41 = 20
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
    USE_SDOT && return _q4_mul_sdot!(Y, A, X)
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

function _q4_mul_sdot!(Y::StridedVecOrMat{Float32}, A::Q4_0Matrix,
        X::StridedVecOrMat{Float32})
    nb = A.nrow ÷ QK
    data = A.data
    T = size(X, 2)
    act = _q8_scratch(A.nrow, T)
    quantize_q8!(act, X)
    GC.@preserve data act begin
        wptr = pointer(data)
        parallel_cols(A.ncol) do lo, hi
            @inbounds for j in lo:hi
                colbase = (j - 1) * nb * BLOCK
                for t in 1:T
                    Y[j, t] = _q4_sdot_col(wptr, colbase, nb,
                        pointer(act.q, (t - 1) * A.nrow + 1), view(act.d, :, t))
                end
            end
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


# ---- q4_1 ---------------------------------------------------------------------
#
# 32 weights/block, 20 bytes: f16 scale d + f16 min m + 16 nibble bytes;
# w = d * q + m. llama.cpp's mixed-quant recipes bump a few tensors to
# q4_1 inside "q4_0" files (e.g. OLMoE's first-layer ffn_down_exps); they
# carry a small share of the math, so this type has the portable f32
# paths only — no SDOT, no device kernel.

struct Q4_1Matrix{D <: AbstractVector{UInt8}} <: AbstractMatrix{Float32}
    data::D
    nrow::Int
    ncol::Int
    function Q4_1Matrix(data::D, nrow::Integer, ncol::Integer) where {D}
        nrow % QK == 0 || error("q4_1 rows must be a multiple of $QK")
        length(data) == (nrow ÷ QK) * BLOCK41 * ncol ||
            error("q4_1 data size mismatch")
        new{D}(data, Int(nrow), Int(ncol))
    end
end

Base.size(A::Q4_1Matrix) = (A.nrow, A.ncol)

Base.@propagate_inbounds function Base.getindex(A::Q4_1Matrix, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    b, p = divrem(i - 1, QK)
    off = ((j - 1) * (A.nrow ÷ QK) + b) * BLOCK41
    q = A.data[off + 4 + (p % 16) + 1]
    nib = p < 16 ? (q & 0x0f) : (q >> 4)
    return _scale(A.data, off) * Float32(nib) + _scale(A.data, off + 2)
end

function LinearAlgebra.mul!(Y::StridedVecOrMat{Float32},
        At::Adjoint{Float32, <:Q4_1Matrix}, X::StridedVecOrMat{Float32})
    A = parent(At)
    size(X, 1) == A.nrow || throw(DimensionMismatch("q4_1 mul"))
    _bytes_backend(A.data) isa KernelAbstractions.CPU ||
        error("q4_1 has no device kernel (host-only tensor type)")
    nb = A.nrow ÷ QK
    data = A.data
    T = size(X, 2)
    Threads.@threads for j in 1:A.ncol
        colbase = (j - 1) * nb * BLOCK41
        for t in 1:T
            acc = 0.0f0
            @inbounds for b in 0:(nb - 1)
                off = colbase + b * BLOCK41
                s = 0.0f0
                sx = 0.0f0
                xoff = b * QK
                @simd for i in 1:16
                    q = data[off + 4 + i]
                    s = muladd(Float32(q & 0x0f), X[xoff + i, t], s)
                    s = muladd(Float32(q >> 4), X[xoff + 16 + i, t], s)
                    sx += X[xoff + i, t] + X[xoff + 16 + i, t]
                end
                acc = muladd(_scale(data, off), s,
                    muladd(_scale(data, off + 2), sx, acc))
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
    USE_SDOT && return _q6_mul_sdot!(Y, A, X)
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

# q6_K · q8 activations: each 16-element sub-scale group is exactly one
# SDOT; the two groups sharing an activation block combine under that
# block's d_a, then the superblock's f16 super-scale
@inline function _q6_sdot_col(wptr::Ptr{UInt8}, off0::Int, nsb::Int,
        qptr::Ptr{Int8}, dact::AbstractVector{Float32})
    tot = 0.0f0
    thirtytwo = Vec{16, Int8}(Int8(32))
    zero4 = Vec{4, Int32}(0)
    @inbounds for sb in 0:(nsb - 1)
        off = off0 + sb * BLOCK6
        ql = wptr + off
        qh = wptr + off + 128
        sc = Ptr{Int8}(wptr + off + 192)
        d = Float32(reinterpret(Float16,
            UInt16(unsafe_load(wptr + off + 208)) |
            (UInt16(unsafe_load(wptr + off + 209)) << 8)))
        sbacc = 0.0f0
        for h in 0:1, g in 0:3
            blockacc = Int32(0)
            for c in 0:1
                vql = vload(Vec{16, UInt8}, ql + h * 64 + (g % 2) * 32 + c * 16)
                vqh = vload(Vec{16, UInt8}, qh + h * 32 + c * 16)
                lo = g < 2 ? vql & 0x0f : vql >> 4
                w = reinterpret(Vec{16, Int8},
                    lo | (((vqh >> (2 * g)) & 0x03) << 4)) - thirtytwo
                a = vload(Vec{16, Int8},
                    qptr + sb * QK6 + h * 128 + g * 32 + c * 16)
                s = sum(sdot(zero4, w, a))
                blockacc += Int32(unsafe_load(sc, h * 8 + g * 2 + c + 1)) * s
            end
            sbacc = muladd(Float32(blockacc),
                dact[sb * 8 + h * 4 + g + 1], sbacc)
        end
        tot = muladd(d, sbacc, tot)
    end
    return tot
end

function _q6_mul_sdot!(Y::StridedVecOrMat{Float32}, A::Q6_KMatrix,
        X::StridedVecOrMat{Float32})
    nsb = A.nrow ÷ QK6
    data = A.data
    T = size(X, 2)
    act = _q8_scratch(A.nrow, T)
    quantize_q8!(act, X)
    GC.@preserve data act begin
        wptr = pointer(data)
        parallel_cols(A.ncol) do lo, hi
            @inbounds for j in lo:hi
                colbase = (j - 1) * nsb * BLOCK6
                for t in 1:T
                    Y[j, t] = _q6_sdot_col(wptr, colbase, nsb,
                        pointer(act.q, (t - 1) * A.nrow + 1), view(act.d, :, t))
                end
            end
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

# device-safe dot of one q4_0 column (at byte offset off0) with X[:, t] —
# the building block for both the generic matmul kernel and callers that
# address columns inside a larger buffer (e.g. fused MoE expert stacks)
@inline function q4_dot(data, off0::Int, X, t::Int, nb::Int)
    acc = 0.0f0
    @inbounds for b in 0:(nb - 1)
        off = off0 + b * BLOCK
        s = 0.0f0
        xoff = b * QK
        for i in 1:16
            q = data[off + 2 + i]
            s = muladd(Float32(q & 0x0f) - 8.0f0, X[xoff + i, t], s)
            s = muladd(Float32(q >> 4) - 8.0f0, X[xoff + 16 + i, t], s)
        end
        acc = muladd(_scale(data, off), s, acc)
    end
    return acc
end

@kernel function _q4_mul_kernel!(Y, @Const(data), @Const(X), nb::Int32)
    j, t = @index(Global, NTuple)
    @inbounds Y[j, t] = q4_dot(data, (j - 1) * Int(nb) * BLOCK, X, t, Int(nb))
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

const AnyQuant = Union{Q4_0Matrix, Q4_1Matrix, Q8_0Matrix, Q6_KMatrix}

# activations and caches allocated "like" quantized weights are plain arrays
Base.similar(A::AnyQuant, ::Type{T}, dims::Dims) where {T} = Array{T}(undef, dims)

KernelAbstractions.get_backend(A::AnyQuant) = _bytes_backend(A.data)

# device movement: MATERIALIZE the bytes before adapting — adapting an mmap
# view directly would rebuild the view over an adapted parent, i.e. upload
# the entire multi-GB file once per tensor
Adapt.adapt_structure(to, A::Q4_0Matrix) =
    Q4_0Matrix(Adapt.adapt(to, collect(A.data)), A.nrow, A.ncol)
Adapt.adapt_structure(to, A::Q4_1Matrix) =
    Q4_1Matrix(Adapt.adapt(to, collect(A.data)), A.nrow, A.ncol)
Adapt.adapt_structure(to, A::Q8_0Matrix) =
    Q8_0Matrix(Adapt.adapt(to, collect(A.data)), A.nrow, A.ncol)
Adapt.adapt_structure(to, A::Q6_KMatrix) =
    Q6_KMatrix(Adapt.adapt(to, collect(A.data)), A.nrow, A.ncol)

# ---- expert stacks ------------------------------------------------------------
#
# A 3-D q4_0 tensor (n_expert weight matrices) kept as ONE buffer, so a
# fused-MoE kernel can address any expert by byte stride, and so device
# movement is one upload per stack instead of n_expert. Indexing yields a
# zero-copy Q4_0Matrix view of one expert for the loop-based (CPU) paths.

struct Q4_0Stack{D <: AbstractVector{UInt8}}
    data::D
    nrow::Int
    ncol::Int
    nexp::Int
end

perexp(A::Q4_0Stack) = (A.nrow ÷ QK) * BLOCK * A.ncol
Base.length(A::Q4_0Stack) = A.nexp
Base.getindex(A::Q4_0Stack, e::Integer) = Q4_0Matrix(
    view(A.data, ((e - 1) * perexp(A) + 1):(e * perexp(A))), A.nrow, A.ncol)

Adapt.adapt_structure(to, A::Q4_0Stack) =
    Q4_0Stack(Adapt.adapt(to, collect(A.data)), A.nrow, A.ncol, A.nexp)

end # module Quant
