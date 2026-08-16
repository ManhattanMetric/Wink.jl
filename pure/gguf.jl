# A pure-Julia GGUF reader: header, key-value metadata, tensor table, and
# zero-copy mmap'd tensor data. No dependencies beyond stdlib. Format
# reference: ggml's gguf.h (v3).
#
# Scope: the unquantized types the pure backend consumes (F32, F16, BF16).
# Quantized blocks parse in the table but materialize lazily only if a
# dequantizer is registered for them — this spike has none.

module GGUF

using Mmap

export GGUFFile, tensor, metadata

const MAGIC = 0x46554747   # "GGUF" little-endian

# ggml tensor types we materialize
const T_F32 = UInt32(0)
const T_F16 = UInt32(1)
const T_BF16 = UInt32(30)

# metadata value types
@enum MetaType::UInt32 begin
    M_U8 = 0
    M_I8 = 1
    M_U16 = 2
    M_I16 = 3
    M_U32 = 4
    M_I32 = 5
    M_F32 = 6
    M_BOOL = 7
    M_STRING = 8
    M_ARRAY = 9
    M_U64 = 10
    M_I64 = 11
    M_F64 = 12
end

struct TensorInfo
    name::String
    dims::Vector{Int}      # ggml order: dims[1] is contiguous/fastest
    typ::UInt32
    offset::UInt64         # from start of data section
end

struct GGUFFile
    path::String
    version::UInt32
    meta::Dict{String, Any}
    tensors::Dict{String, TensorInfo}
    data::Vector{UInt8}    # mmap of the aligned data section
end

read_str(io) = String(read(io, read(io, UInt64)))

function read_value(io, t::MetaType)
    t == M_U8 && return read(io, UInt8)
    t == M_I8 && return read(io, Int8)
    t == M_U16 && return read(io, UInt16)
    t == M_I16 && return read(io, Int16)
    t == M_U32 && return read(io, UInt32)
    t == M_I32 && return read(io, Int32)
    t == M_F32 && return read(io, Float32)
    t == M_BOOL && return read(io, UInt8) != 0
    t == M_STRING && return read_str(io)
    t == M_U64 && return read(io, UInt64)
    t == M_I64 && return read(io, Int64)
    t == M_F64 && return read(io, Float64)
    if t == M_ARRAY
        et = MetaType(read(io, UInt32))
        n = read(io, UInt64)
        return [read_value(io, et) for _ in 1:n]
    end
    error("unknown GGUF metadata type $t")
end

"""
    GGUFFile(path) -> GGUFFile

Parse a GGUF file's metadata and tensor table; mmap the data section.
"""
function GGUFFile(path::AbstractString)
    open(path, "r") do io
        read(io, UInt32) == MAGIC || error("not a GGUF file: $path")
        version = read(io, UInt32)
        version in (2, 3) || error("unsupported GGUF version $version")
        n_tensors = Int(read(io, UInt64))
        n_kv = Int(read(io, UInt64))
        meta = Dict{String, Any}()
        for _ in 1:n_kv
            key = read_str(io)
            meta[key] = read_value(io, MetaType(read(io, UInt32)))
        end
        infos = Vector{TensorInfo}(undef, n_tensors)
        for i in 1:n_tensors
            name = read_str(io)
            nd = Int(read(io, UInt32))
            dims = [Int(read(io, UInt64)) for _ in 1:nd]
            typ = read(io, UInt32)
            off = read(io, UInt64)
            infos[i] = TensorInfo(name, dims, typ, off)
        end
        alignment = Int(get(meta, "general.alignment", UInt32(32)))
        pos = position(io)
        data_start = cld(pos, alignment) * alignment
        data = Mmap.mmap(io, Vector{UInt8}, filesize(path) - data_start,
            data_start)
        return GGUFFile(String(path), version,
            meta, Dict(t.name => t for t in infos), data)
    end
end

metadata(f::GGUFFile, key::AbstractString) = f.meta[key]
metadata(f::GGUFFile, key::AbstractString, default) = get(f.meta, key, default)

# bf16 -> f32: the top 16 bits of an f32
bf16_to_f32(u::UInt16) = reinterpret(Float32, UInt32(u) << 16)

"""
    tensor(f, name; T = Float32) -> Array{T}

Materialize a tensor as a Julia array in ggml dimension order (first axis
contiguous). F32/F16/BF16 only — this backend runs on unquantized weights.
"""
function tensor(f::GGUFFile, name::AbstractString; T::Type = Float32)
    ti = get(f.tensors, name, nothing)
    ti === nothing && error("no tensor named $name (have $(length(f.tensors)))")
    n = prod(ti.dims)
    off = Int(ti.offset)
    raw = if ti.typ == T_F32
        reinterpret(Float32, view(f.data, (off + 1):(off + 4n)))
    elseif ti.typ == T_F16
        reinterpret(Float16, view(f.data, (off + 1):(off + 2n)))
    elseif ti.typ == T_BF16
        bf16_to_f32.(reinterpret(UInt16, view(f.data, (off + 1):(off + 2n))))
    else
        error("tensor $name has unsupported ggml type $(ti.typ) — " *
              "unquantized (F16/BF16/F32) GGUFs only")
    end
    return reshape(T.(raw), ti.dims...)
end

end # module GGUF
