module LibLlama

using CEnum: CEnum, @cenum

# Prologue injected at the top of the generated LibLlama.jl: point the
# bindings at the vendored dylib (its @loader_path rpath resolves the ggml
# dependencies sitting beside it).
const libllama = joinpath(@__DIR__, "vendor", "llama-b10405", "libllama.dylib")


# typedef void ( * ggml_abort_callback_t ) ( const char * error_message )
const ggml_abort_callback_t = Ptr{Cvoid}

function ggml_set_abort_callback(callback)
    ccall((:ggml_set_abort_callback, libllama), ggml_abort_callback_t, (ggml_abort_callback_t,), callback)
end

@cenum ggml_status::Int32 begin
    GGML_STATUS_ALLOC_FAILED = -2
    GGML_STATUS_FAILED = -1
    GGML_STATUS_SUCCESS = 0
    GGML_STATUS_ABORTED = 1
end

function ggml_status_to_string(status)
    ccall((:ggml_status_to_string, libllama), Ptr{Cchar}, (ggml_status,), status)
end

const ggml_fp16_t = UInt16

function ggml_fp16_to_fp32(arg1)
    ccall((:ggml_fp16_to_fp32, libllama), Cfloat, (ggml_fp16_t,), arg1)
end

function ggml_fp32_to_fp16(arg1)
    ccall((:ggml_fp32_to_fp16, libllama), ggml_fp16_t, (Cfloat,), arg1)
end

function ggml_fp16_to_fp32_row(arg1, arg2, arg3)
    ccall((:ggml_fp16_to_fp32_row, libllama), Cvoid, (Ptr{ggml_fp16_t}, Ptr{Cfloat}, Int64), arg1, arg2, arg3)
end

function ggml_fp32_to_fp16_row(arg1, arg2, arg3)
    ccall((:ggml_fp32_to_fp16_row, libllama), Cvoid, (Ptr{Cfloat}, Ptr{ggml_fp16_t}, Int64), arg1, arg2, arg3)
end

struct ggml_bf16_t
    bits::UInt16
end

function ggml_fp32_to_bf16(arg1)
    ccall((:ggml_fp32_to_bf16, libllama), ggml_bf16_t, (Cfloat,), arg1)
end

function ggml_bf16_to_fp32(arg1)
    ccall((:ggml_bf16_to_fp32, libllama), Cfloat, (ggml_bf16_t,), arg1)
end

function ggml_bf16_to_fp32_row(arg1, arg2, arg3)
    ccall((:ggml_bf16_to_fp32_row, libllama), Cvoid, (Ptr{ggml_bf16_t}, Ptr{Cfloat}, Int64), arg1, arg2, arg3)
end

function ggml_fp32_to_bf16_row_ref(arg1, arg2, arg3)
    ccall((:ggml_fp32_to_bf16_row_ref, libllama), Cvoid, (Ptr{Cfloat}, Ptr{ggml_bf16_t}, Int64), arg1, arg2, arg3)
end

function ggml_fp32_to_bf16_row(arg1, arg2, arg3)
    ccall((:ggml_fp32_to_bf16_row, libllama), Cvoid, (Ptr{Cfloat}, Ptr{ggml_bf16_t}, Int64), arg1, arg2, arg3)
end

mutable struct ggml_object end

@cenum ggml_type::UInt32 begin
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    GGML_TYPE_Q5_0 = 6
    GGML_TYPE_Q5_1 = 7
    GGML_TYPE_Q8_0 = 8
    GGML_TYPE_Q8_1 = 9
    GGML_TYPE_Q2_K = 10
    GGML_TYPE_Q3_K = 11
    GGML_TYPE_Q4_K = 12
    GGML_TYPE_Q5_K = 13
    GGML_TYPE_Q6_K = 14
    GGML_TYPE_Q8_K = 15
    GGML_TYPE_IQ2_XXS = 16
    GGML_TYPE_IQ2_XS = 17
    GGML_TYPE_IQ3_XXS = 18
    GGML_TYPE_IQ1_S = 19
    GGML_TYPE_IQ4_NL = 20
    GGML_TYPE_IQ3_S = 21
    GGML_TYPE_IQ2_S = 22
    GGML_TYPE_IQ4_XS = 23
    GGML_TYPE_I8 = 24
    GGML_TYPE_I16 = 25
    GGML_TYPE_I32 = 26
    GGML_TYPE_I64 = 27
    GGML_TYPE_F64 = 28
    GGML_TYPE_IQ1_M = 29
    GGML_TYPE_BF16 = 30
    GGML_TYPE_TQ1_0 = 34
    GGML_TYPE_TQ2_0 = 35
    GGML_TYPE_MXFP4 = 39
    GGML_TYPE_NVFP4 = 40
    GGML_TYPE_Q1_0 = 41
    GGML_TYPE_Q2_0 = 42
    GGML_TYPE_COUNT = 43
end

@cenum ggml_prec::UInt32 begin
    GGML_PREC_DEFAULT = 0
    GGML_PREC_F32 = 10
end

@cenum ggml_op_hint::UInt32 begin
    GGML_HINT_NONE = 0
    GGML_HINT_SRC0_IS_HADAMARD = 1
end

@cenum ggml_ftype::Int32 begin
    GGML_FTYPE_UNKNOWN = -1
    GGML_FTYPE_ALL_F32 = 0
    GGML_FTYPE_MOSTLY_F16 = 1
    GGML_FTYPE_MOSTLY_Q4_0 = 2
    GGML_FTYPE_MOSTLY_Q4_1 = 3
    GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4
    GGML_FTYPE_MOSTLY_Q8_0 = 7
    GGML_FTYPE_MOSTLY_Q5_0 = 8
    GGML_FTYPE_MOSTLY_Q5_1 = 9
    GGML_FTYPE_MOSTLY_Q2_K = 10
    GGML_FTYPE_MOSTLY_Q3_K = 11
    GGML_FTYPE_MOSTLY_Q4_K = 12
    GGML_FTYPE_MOSTLY_Q5_K = 13
    GGML_FTYPE_MOSTLY_Q6_K = 14
    GGML_FTYPE_MOSTLY_IQ2_XXS = 15
    GGML_FTYPE_MOSTLY_IQ2_XS = 16
    GGML_FTYPE_MOSTLY_IQ3_XXS = 17
    GGML_FTYPE_MOSTLY_IQ1_S = 18
    GGML_FTYPE_MOSTLY_IQ4_NL = 19
    GGML_FTYPE_MOSTLY_IQ3_S = 20
    GGML_FTYPE_MOSTLY_IQ2_S = 21
    GGML_FTYPE_MOSTLY_IQ4_XS = 22
    GGML_FTYPE_MOSTLY_IQ1_M = 23
    GGML_FTYPE_MOSTLY_BF16 = 24
    GGML_FTYPE_MOSTLY_MXFP4 = 25
    GGML_FTYPE_MOSTLY_NVFP4 = 26
    GGML_FTYPE_MOSTLY_Q1_0 = 27
    GGML_FTYPE_MOSTLY_Q2_0 = 28
end

@cenum ggml_op::UInt32 begin
    GGML_OP_NONE = 0
    GGML_OP_DUP = 1
    GGML_OP_ADD = 2
    GGML_OP_ADD_ID = 3
    GGML_OP_ADD1 = 4
    GGML_OP_ACC = 5
    GGML_OP_SUB = 6
    GGML_OP_MUL = 7
    GGML_OP_DIV = 8
    GGML_OP_SQR = 9
    GGML_OP_SQRT = 10
    GGML_OP_LOG = 11
    GGML_OP_SIN = 12
    GGML_OP_COS = 13
    GGML_OP_SUM = 14
    GGML_OP_SUM_ROWS = 15
    GGML_OP_CUMSUM = 16
    GGML_OP_MEAN = 17
    GGML_OP_ARGMAX = 18
    GGML_OP_COUNT_EQUAL = 19
    GGML_OP_REPEAT = 20
    GGML_OP_REPEAT_BACK = 21
    GGML_OP_CONCAT = 22
    GGML_OP_SILU_BACK = 23
    GGML_OP_NORM = 24
    GGML_OP_RMS_NORM = 25
    GGML_OP_RMS_NORM_BACK = 26
    GGML_OP_GROUP_NORM = 27
    GGML_OP_L2_NORM = 28
    GGML_OP_MUL_MAT = 29
    GGML_OP_MUL_MAT_ID = 30
    GGML_OP_OUT_PROD = 31
    GGML_OP_SCALE = 32
    GGML_OP_SET = 33
    GGML_OP_CPY = 34
    GGML_OP_CONT = 35
    GGML_OP_RESHAPE = 36
    GGML_OP_VIEW = 37
    GGML_OP_PERMUTE = 38
    GGML_OP_TRANSPOSE = 39
    GGML_OP_GET_ROWS = 40
    GGML_OP_GET_ROWS_BACK = 41
    GGML_OP_SET_ROWS = 42
    GGML_OP_DIAG = 43
    GGML_OP_DIAG_MASK_INF = 44
    GGML_OP_DIAG_MASK_ZERO = 45
    GGML_OP_SOFT_MAX = 46
    GGML_OP_SOFT_MAX_BACK = 47
    GGML_OP_ROPE = 48
    GGML_OP_ROPE_BACK = 49
    GGML_OP_CLAMP = 50
    GGML_OP_CONV_TRANSPOSE_1D = 51
    GGML_OP_IM2COL = 52
    GGML_OP_IM2COL_BACK = 53
    GGML_OP_IM2COL_3D = 54
    GGML_OP_COL2IM_1D = 55
    GGML_OP_CONV_2D = 56
    GGML_OP_CONV_3D = 57
    GGML_OP_CONV_2D_DW = 58
    GGML_OP_CONV_TRANSPOSE_2D = 59
    GGML_OP_POOL_1D = 60
    GGML_OP_POOL_2D = 61
    GGML_OP_POOL_2D_BACK = 62
    GGML_OP_UPSCALE = 63
    GGML_OP_PAD = 64
    GGML_OP_PAD_REFLECT_1D = 65
    GGML_OP_ROLL = 66
    GGML_OP_ARANGE = 67
    GGML_OP_TIMESTEP_EMBEDDING = 68
    GGML_OP_ARGSORT = 69
    GGML_OP_TOP_K = 70
    GGML_OP_LEAKY_RELU = 71
    GGML_OP_TRI = 72
    GGML_OP_FILL = 73
    GGML_OP_FLASH_ATTN_EXT = 74
    GGML_OP_FLASH_ATTN_BACK = 75
    GGML_OP_SSM_CONV = 76
    GGML_OP_SSM_SCAN = 77
    GGML_OP_WIN_PART = 78
    GGML_OP_WIN_UNPART = 79
    GGML_OP_GET_REL_POS = 80
    GGML_OP_ADD_REL_POS = 81
    GGML_OP_RWKV_WKV6 = 82
    GGML_OP_GATED_LINEAR_ATTN = 83
    GGML_OP_RWKV_WKV7 = 84
    GGML_OP_SOLVE_TRI = 85
    GGML_OP_GATED_DELTA_NET = 86
    GGML_OP_LIGHTNING_INDEXER = 87
    GGML_OP_DSV4_HC_COMB = 88
    GGML_OP_DSV4_HC_PRE = 89
    GGML_OP_DSV4_HC_POST = 90
    GGML_OP_UNARY = 91
    GGML_OP_MAP_CUSTOM1 = 92
    GGML_OP_MAP_CUSTOM2 = 93
    GGML_OP_MAP_CUSTOM3 = 94
    GGML_OP_CUSTOM = 95
    GGML_OP_CROSS_ENTROPY_LOSS = 96
    GGML_OP_CROSS_ENTROPY_LOSS_BACK = 97
    GGML_OP_OPT_STEP_ADAMW = 98
    GGML_OP_OPT_STEP_SGD = 99
    GGML_OP_GLU = 100
    GGML_OP_COUNT = 101
end

@cenum ggml_unary_op::UInt32 begin
    GGML_UNARY_OP_ABS = 0
    GGML_UNARY_OP_SGN = 1
    GGML_UNARY_OP_NEG = 2
    GGML_UNARY_OP_STEP = 3
    GGML_UNARY_OP_TANH = 4
    GGML_UNARY_OP_ELU = 5
    GGML_UNARY_OP_RELU = 6
    GGML_UNARY_OP_SIGMOID = 7
    GGML_UNARY_OP_GELU = 8
    GGML_UNARY_OP_GELU_QUICK = 9
    GGML_UNARY_OP_SILU = 10
    GGML_UNARY_OP_HARDSWISH = 11
    GGML_UNARY_OP_HARDSIGMOID = 12
    GGML_UNARY_OP_EXP = 13
    GGML_UNARY_OP_EXPM1 = 14
    GGML_UNARY_OP_SOFTPLUS = 15
    GGML_UNARY_OP_GELU_ERF = 16
    GGML_UNARY_OP_XIELU = 17
    GGML_UNARY_OP_FLOOR = 18
    GGML_UNARY_OP_CEIL = 19
    GGML_UNARY_OP_ROUND = 20
    GGML_UNARY_OP_TRUNC = 21
    GGML_UNARY_OP_COUNT = 22
end

@cenum ggml_glu_op::UInt32 begin
    GGML_GLU_OP_REGLU = 0
    GGML_GLU_OP_GEGLU = 1
    GGML_GLU_OP_SWIGLU = 2
    GGML_GLU_OP_SWIGLU_OAI = 3
    GGML_GLU_OP_GEGLU_ERF = 4
    GGML_GLU_OP_GEGLU_QUICK = 5
    GGML_GLU_OP_COUNT = 6
end

@cenum ggml_object_type::UInt32 begin
    GGML_OBJECT_TYPE_TENSOR = 0
    GGML_OBJECT_TYPE_GRAPH = 1
    GGML_OBJECT_TYPE_WORK_BUFFER = 2
end

@cenum ggml_log_level::UInt32 begin
    GGML_LOG_LEVEL_NONE = 0
    GGML_LOG_LEVEL_DEBUG = 1
    GGML_LOG_LEVEL_INFO = 2
    GGML_LOG_LEVEL_WARN = 3
    GGML_LOG_LEVEL_ERROR = 4
    GGML_LOG_LEVEL_CONT = 5
end

@cenum ggml_tensor_flag::UInt32 begin
    GGML_TENSOR_FLAG_INPUT = 1
    GGML_TENSOR_FLAG_OUTPUT = 2
    GGML_TENSOR_FLAG_PARAM = 4
    GGML_TENSOR_FLAG_LOSS = 8
    GGML_TENSOR_FLAG_COMPUTE = 16
end

@cenum ggml_tri_type::UInt32 begin
    GGML_TRI_TYPE_UPPER_DIAG = 0
    GGML_TRI_TYPE_UPPER = 1
    GGML_TRI_TYPE_LOWER_DIAG = 2
    GGML_TRI_TYPE_LOWER = 3
end

struct ggml_init_params
    mem_size::Csize_t
    mem_buffer::Ptr{Cvoid}
    no_alloc::Bool
end

mutable struct ggml_backend_buffer end

struct ggml_tensor
    type::ggml_type
    buffer::Ptr{ggml_backend_buffer}
    ne::NTuple{4, Int64}
    nb::NTuple{4, Csize_t}
    op::ggml_op
    op_params::NTuple{16, Int32}
    flags::Int32
    src::NTuple{10, Ptr{ggml_tensor}}
    view_src::Ptr{ggml_tensor}
    view_offs::Csize_t
    data::Ptr{Cvoid}
    name::NTuple{64, Cchar}
    extra::Ptr{Cvoid}
    padding::NTuple{8, Cchar}
end

# typedef bool ( * ggml_abort_callback ) ( void * data )
const ggml_abort_callback = Ptr{Cvoid}

const ggml_guid = NTuple{16, UInt8}

const ggml_guid_t = Ptr{ggml_guid}

function ggml_guid_matches(guid_a, guid_b)
    ccall((:ggml_guid_matches, libllama), Bool, (ggml_guid_t, ggml_guid_t), guid_a, guid_b)
end

function ggml_version()
    ccall((:ggml_version, libllama), Ptr{Cchar}, ())
end

function ggml_commit()
    ccall((:ggml_commit, libllama), Ptr{Cchar}, ())
end

function ggml_time_init()
    ccall((:ggml_time_init, libllama), Cvoid, ())
end

function ggml_time_ms()
    ccall((:ggml_time_ms, libllama), Int64, ())
end

function ggml_time_us()
    ccall((:ggml_time_us, libllama), Int64, ())
end

function ggml_cycles()
    ccall((:ggml_cycles, libllama), Int64, ())
end

function ggml_cycles_per_ms()
    ccall((:ggml_cycles_per_ms, libllama), Int64, ())
end

function ggml_fopen(fname, mode)
    ccall((:ggml_fopen, libllama), Ptr{Libc.FILE}, (Ptr{Cchar}, Ptr{Cchar}), fname, mode)
end

function ggml_print_object(obj)
    ccall((:ggml_print_object, libllama), Cvoid, (Ptr{ggml_object},), obj)
end

mutable struct ggml_context end

function ggml_print_objects(ctx)
    ccall((:ggml_print_objects, libllama), Cvoid, (Ptr{ggml_context},), ctx)
end

function ggml_nelements(tensor)
    ccall((:ggml_nelements, libllama), Int64, (Ptr{ggml_tensor},), tensor)
end

function ggml_nrows(tensor)
    ccall((:ggml_nrows, libllama), Int64, (Ptr{ggml_tensor},), tensor)
end

function ggml_nbytes(tensor)
    ccall((:ggml_nbytes, libllama), Csize_t, (Ptr{ggml_tensor},), tensor)
end

function ggml_nbytes_pad(tensor)
    ccall((:ggml_nbytes_pad, libllama), Csize_t, (Ptr{ggml_tensor},), tensor)
end

function ggml_blck_size(type)
    ccall((:ggml_blck_size, libllama), Int64, (ggml_type,), type)
end

function ggml_type_size(type)
    ccall((:ggml_type_size, libllama), Csize_t, (ggml_type,), type)
end

function ggml_row_size(type, ne)
    ccall((:ggml_row_size, libllama), Csize_t, (ggml_type, Int64), type, ne)
end

function ggml_type_sizef(type)
    ccall((:ggml_type_sizef, libllama), Cdouble, (ggml_type,), type)
end

function ggml_type_name(type)
    ccall((:ggml_type_name, libllama), Ptr{Cchar}, (ggml_type,), type)
end

function ggml_op_name(op)
    ccall((:ggml_op_name, libllama), Ptr{Cchar}, (ggml_op,), op)
end

function ggml_op_symbol(op)
    ccall((:ggml_op_symbol, libllama), Ptr{Cchar}, (ggml_op,), op)
end

function ggml_unary_op_name(op)
    ccall((:ggml_unary_op_name, libllama), Ptr{Cchar}, (ggml_unary_op,), op)
end

function ggml_glu_op_name(op)
    ccall((:ggml_glu_op_name, libllama), Ptr{Cchar}, (ggml_glu_op,), op)
end

function ggml_op_desc(t)
    ccall((:ggml_op_desc, libllama), Ptr{Cchar}, (Ptr{ggml_tensor},), t)
end

function ggml_element_size(tensor)
    ccall((:ggml_element_size, libllama), Csize_t, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_quantized(type)
    ccall((:ggml_is_quantized, libllama), Bool, (ggml_type,), type)
end

function ggml_ftype_to_ggml_type(ftype)
    ccall((:ggml_ftype_to_ggml_type, libllama), ggml_type, (ggml_ftype,), ftype)
end

function ggml_is_transposed(tensor)
    ccall((:ggml_is_transposed, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_permuted(tensor)
    ccall((:ggml_is_permuted, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_empty(tensor)
    ccall((:ggml_is_empty, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_view(tensor)
    ccall((:ggml_is_view, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_scalar(tensor)
    ccall((:ggml_is_scalar, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_vector(tensor)
    ccall((:ggml_is_vector, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_matrix(tensor)
    ccall((:ggml_is_matrix, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_3d(tensor)
    ccall((:ggml_is_3d, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_n_dims(tensor)
    ccall((:ggml_n_dims, libllama), Cint, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_contiguous(tensor)
    ccall((:ggml_is_contiguous, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_contiguous_0(tensor)
    ccall((:ggml_is_contiguous_0, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_contiguous_1(tensor)
    ccall((:ggml_is_contiguous_1, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_contiguous_2(tensor)
    ccall((:ggml_is_contiguous_2, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_contiguous_to_1(tensor)
    ccall((:ggml_is_contiguous_to_1, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_contiguous_to_2(tensor)
    ccall((:ggml_is_contiguous_to_2, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_contiguous_to_3(tensor)
    ccall((:ggml_is_contiguous_to_3, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_contiguously_allocated(tensor)
    ccall((:ggml_is_contiguously_allocated, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_contiguous_channels(tensor)
    ccall((:ggml_is_contiguous_channels, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_is_contiguous_rows(tensor)
    ccall((:ggml_is_contiguous_rows, libllama), Bool, (Ptr{ggml_tensor},), tensor)
end

function ggml_are_same_shape(t0, t1)
    ccall((:ggml_are_same_shape, libllama), Bool, (Ptr{ggml_tensor}, Ptr{ggml_tensor}), t0, t1)
end

function ggml_are_same_stride(t0, t1)
    ccall((:ggml_are_same_stride, libllama), Bool, (Ptr{ggml_tensor}, Ptr{ggml_tensor}), t0, t1)
end

function ggml_can_repeat(t0, t1)
    ccall((:ggml_can_repeat, libllama), Bool, (Ptr{ggml_tensor}, Ptr{ggml_tensor}), t0, t1)
end

function ggml_tensor_overhead()
    ccall((:ggml_tensor_overhead, libllama), Csize_t, ())
end

function ggml_validate_row_data(type, data, nbytes)
    ccall((:ggml_validate_row_data, libllama), Bool, (ggml_type, Ptr{Cvoid}, Csize_t), type, data, nbytes)
end

function ggml_init(params)
    ccall((:ggml_init, libllama), Ptr{ggml_context}, (ggml_init_params,), params)
end

function ggml_reset(ctx)
    ccall((:ggml_reset, libllama), Cvoid, (Ptr{ggml_context},), ctx)
end

function ggml_free(ctx)
    ccall((:ggml_free, libllama), Cvoid, (Ptr{ggml_context},), ctx)
end

function ggml_used_mem(ctx)
    ccall((:ggml_used_mem, libllama), Csize_t, (Ptr{ggml_context},), ctx)
end

function ggml_get_no_alloc(ctx)
    ccall((:ggml_get_no_alloc, libllama), Bool, (Ptr{ggml_context},), ctx)
end

function ggml_set_no_alloc(ctx, no_alloc)
    ccall((:ggml_set_no_alloc, libllama), Cvoid, (Ptr{ggml_context}, Bool), ctx, no_alloc)
end

function ggml_get_mem_buffer(ctx)
    ccall((:ggml_get_mem_buffer, libllama), Ptr{Cvoid}, (Ptr{ggml_context},), ctx)
end

function ggml_get_mem_size(ctx)
    ccall((:ggml_get_mem_size, libllama), Csize_t, (Ptr{ggml_context},), ctx)
end

function ggml_get_max_tensor_size(ctx)
    ccall((:ggml_get_max_tensor_size, libllama), Csize_t, (Ptr{ggml_context},), ctx)
end

function ggml_new_tensor(ctx, type, n_dims, ne)
    ccall((:ggml_new_tensor, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, ggml_type, Cint, Ptr{Int64}), ctx, type, n_dims, ne)
end

function ggml_new_tensor_1d(ctx, type, ne0)
    ccall((:ggml_new_tensor_1d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, ggml_type, Int64), ctx, type, ne0)
end

function ggml_new_tensor_2d(ctx, type, ne0, ne1)
    ccall((:ggml_new_tensor_2d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, ggml_type, Int64, Int64), ctx, type, ne0, ne1)
end

function ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2)
    ccall((:ggml_new_tensor_3d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, ggml_type, Int64, Int64, Int64), ctx, type, ne0, ne1, ne2)
end

function ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3)
    ccall((:ggml_new_tensor_4d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, ggml_type, Int64, Int64, Int64, Int64), ctx, type, ne0, ne1, ne2, ne3)
end

function ggml_new_buffer(ctx, nbytes)
    ccall((:ggml_new_buffer, libllama), Ptr{Cvoid}, (Ptr{ggml_context}, Csize_t), ctx, nbytes)
end

function ggml_dup_tensor(ctx, src)
    ccall((:ggml_dup_tensor, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, src)
end

function ggml_view_tensor(ctx, src)
    ccall((:ggml_view_tensor, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, src)
end

function ggml_get_first_tensor(ctx)
    ccall((:ggml_get_first_tensor, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context},), ctx)
end

function ggml_get_next_tensor(ctx, tensor)
    ccall((:ggml_get_next_tensor, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, tensor)
end

function ggml_get_tensor(ctx, name)
    ccall((:ggml_get_tensor, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{Cchar}), ctx, name)
end

function ggml_unravel_index(tensor, i, i0, i1, i2, i3)
    ccall((:ggml_unravel_index, libllama), Cvoid, (Ptr{ggml_tensor}, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}), tensor, i, i0, i1, i2, i3)
end

function ggml_get_unary_op(tensor)
    ccall((:ggml_get_unary_op, libllama), ggml_unary_op, (Ptr{ggml_tensor},), tensor)
end

function ggml_get_glu_op(tensor)
    ccall((:ggml_get_glu_op, libllama), ggml_glu_op, (Ptr{ggml_tensor},), tensor)
end

function ggml_get_data(tensor)
    ccall((:ggml_get_data, libllama), Ptr{Cvoid}, (Ptr{ggml_tensor},), tensor)
end

function ggml_get_data_f32(tensor)
    ccall((:ggml_get_data_f32, libllama), Ptr{Cfloat}, (Ptr{ggml_tensor},), tensor)
end

function ggml_get_name(tensor)
    ccall((:ggml_get_name, libllama), Ptr{Cchar}, (Ptr{ggml_tensor},), tensor)
end

function ggml_set_name(tensor, name)
    ccall((:ggml_set_name, libllama), Ptr{ggml_tensor}, (Ptr{ggml_tensor}, Ptr{Cchar}), tensor, name)
end

function ggml_set_input(tensor)
    ccall((:ggml_set_input, libllama), Cvoid, (Ptr{ggml_tensor},), tensor)
end

function ggml_set_output(tensor)
    ccall((:ggml_set_output, libllama), Cvoid, (Ptr{ggml_tensor},), tensor)
end

function ggml_set_param(tensor)
    ccall((:ggml_set_param, libllama), Cvoid, (Ptr{ggml_tensor},), tensor)
end

function ggml_set_loss(tensor)
    ccall((:ggml_set_loss, libllama), Cvoid, (Ptr{ggml_tensor},), tensor)
end

function ggml_dup(ctx, a)
    ccall((:ggml_dup, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_dup_inplace(ctx, a)
    ccall((:ggml_dup_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_add(ctx, a, b)
    ccall((:ggml_add, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_add_inplace(ctx, a, b)
    ccall((:ggml_add_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_add_cast(ctx, a, b, type)
    ccall((:ggml_add_cast, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_type), ctx, a, b, type)
end

function ggml_add_id(ctx, a, b, ids)
    ccall((:ggml_add_id, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b, ids)
end

function ggml_add1(ctx, a, b)
    ccall((:ggml_add1, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_add1_inplace(ctx, a, b)
    ccall((:ggml_add1_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_acc(ctx, a, b, nb1, nb2, nb3, offset)
    ccall((:ggml_acc, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t, Csize_t, Csize_t, Csize_t), ctx, a, b, nb1, nb2, nb3, offset)
end

function ggml_acc_inplace(ctx, a, b, nb1, nb2, nb3, offset)
    ccall((:ggml_acc_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t, Csize_t, Csize_t, Csize_t), ctx, a, b, nb1, nb2, nb3, offset)
end

function ggml_sub(ctx, a, b)
    ccall((:ggml_sub, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_sub_inplace(ctx, a, b)
    ccall((:ggml_sub_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_mul(ctx, a, b)
    ccall((:ggml_mul, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_mul_inplace(ctx, a, b)
    ccall((:ggml_mul_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_div(ctx, a, b)
    ccall((:ggml_div, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_div_inplace(ctx, a, b)
    ccall((:ggml_div_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_sqr(ctx, a)
    ccall((:ggml_sqr, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sqr_inplace(ctx, a)
    ccall((:ggml_sqr_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sqrt(ctx, a)
    ccall((:ggml_sqrt, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sqrt_inplace(ctx, a)
    ccall((:ggml_sqrt_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_log(ctx, a)
    ccall((:ggml_log, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_log_inplace(ctx, a)
    ccall((:ggml_log_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_expm1(ctx, a)
    ccall((:ggml_expm1, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_expm1_inplace(ctx, a)
    ccall((:ggml_expm1_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_softplus(ctx, a)
    ccall((:ggml_softplus, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_softplus_inplace(ctx, a)
    ccall((:ggml_softplus_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sin(ctx, a)
    ccall((:ggml_sin, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sin_inplace(ctx, a)
    ccall((:ggml_sin_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_cos(ctx, a)
    ccall((:ggml_cos, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_cos_inplace(ctx, a)
    ccall((:ggml_cos_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sum(ctx, a)
    ccall((:ggml_sum, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sum_rows(ctx, a)
    ccall((:ggml_sum_rows, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_cumsum(ctx, a)
    ccall((:ggml_cumsum, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_mean(ctx, a)
    ccall((:ggml_mean, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_argmax(ctx, a)
    ccall((:ggml_argmax, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_count_equal(ctx, a, b)
    ccall((:ggml_count_equal, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_repeat(ctx, a, b)
    ccall((:ggml_repeat, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_repeat_4d(ctx, a, ne0, ne1, ne2, ne3)
    ccall((:ggml_repeat_4d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64, Int64), ctx, a, ne0, ne1, ne2, ne3)
end

function ggml_repeat_back(ctx, a, b)
    ccall((:ggml_repeat_back, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_concat(ctx, a, b, dim)
    ccall((:ggml_concat, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint), ctx, a, b, dim)
end

function ggml_abs(ctx, a)
    ccall((:ggml_abs, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_abs_inplace(ctx, a)
    ccall((:ggml_abs_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sgn(ctx, a)
    ccall((:ggml_sgn, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sgn_inplace(ctx, a)
    ccall((:ggml_sgn_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_neg(ctx, a)
    ccall((:ggml_neg, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_neg_inplace(ctx, a)
    ccall((:ggml_neg_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_step(ctx, a)
    ccall((:ggml_step, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_step_inplace(ctx, a)
    ccall((:ggml_step_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_tanh(ctx, a)
    ccall((:ggml_tanh, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_tanh_inplace(ctx, a)
    ccall((:ggml_tanh_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_elu(ctx, a)
    ccall((:ggml_elu, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_elu_inplace(ctx, a)
    ccall((:ggml_elu_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_relu(ctx, a)
    ccall((:ggml_relu, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_leaky_relu(ctx, a, negative_slope, inplace)
    ccall((:ggml_leaky_relu, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat, Bool), ctx, a, negative_slope, inplace)
end

function ggml_relu_inplace(ctx, a)
    ccall((:ggml_relu_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sigmoid(ctx, a)
    ccall((:ggml_sigmoid, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_sigmoid_inplace(ctx, a)
    ccall((:ggml_sigmoid_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_gelu(ctx, a)
    ccall((:ggml_gelu, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_gelu_inplace(ctx, a)
    ccall((:ggml_gelu_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_gelu_erf(ctx, a)
    ccall((:ggml_gelu_erf, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_gelu_erf_inplace(ctx, a)
    ccall((:ggml_gelu_erf_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_gelu_quick(ctx, a)
    ccall((:ggml_gelu_quick, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_gelu_quick_inplace(ctx, a)
    ccall((:ggml_gelu_quick_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_silu(ctx, a)
    ccall((:ggml_silu, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_silu_inplace(ctx, a)
    ccall((:ggml_silu_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_silu_back(ctx, a, b)
    ccall((:ggml_silu_back, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_hardswish(ctx, a)
    ccall((:ggml_hardswish, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_hardsigmoid(ctx, a)
    ccall((:ggml_hardsigmoid, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_exp(ctx, a)
    ccall((:ggml_exp, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_exp_inplace(ctx, a)
    ccall((:ggml_exp_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_floor(ctx, a)
    ccall((:ggml_floor, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_floor_inplace(ctx, a)
    ccall((:ggml_floor_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_ceil(ctx, a)
    ccall((:ggml_ceil, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_ceil_inplace(ctx, a)
    ccall((:ggml_ceil_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_round(ctx, a)
    ccall((:ggml_round, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_round_inplace(ctx, a)
    ccall((:ggml_round_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_trunc(ctx, a)
    ccall((:ggml_trunc, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_trunc_inplace(ctx, a)
    ccall((:ggml_trunc_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_xielu(ctx, a, alpha_n, alpha_p, beta, eps)
    ccall((:ggml_xielu, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat, Cfloat, Cfloat, Cfloat), ctx, a, alpha_n, alpha_p, beta, eps)
end

function ggml_glu(ctx, a, op, swapped)
    ccall((:ggml_glu, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_glu_op, Bool), ctx, a, op, swapped)
end

function ggml_reglu(ctx, a)
    ccall((:ggml_reglu, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_reglu_swapped(ctx, a)
    ccall((:ggml_reglu_swapped, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_geglu(ctx, a)
    ccall((:ggml_geglu, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_geglu_swapped(ctx, a)
    ccall((:ggml_geglu_swapped, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_swiglu(ctx, a)
    ccall((:ggml_swiglu, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_swiglu_swapped(ctx, a)
    ccall((:ggml_swiglu_swapped, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_geglu_erf(ctx, a)
    ccall((:ggml_geglu_erf, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_geglu_erf_swapped(ctx, a)
    ccall((:ggml_geglu_erf_swapped, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_geglu_quick(ctx, a)
    ccall((:ggml_geglu_quick, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_geglu_quick_swapped(ctx, a)
    ccall((:ggml_geglu_quick_swapped, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_glu_split(ctx, a, b, op)
    ccall((:ggml_glu_split, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_glu_op), ctx, a, b, op)
end

function ggml_reglu_split(ctx, a, b)
    ccall((:ggml_reglu_split, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_geglu_split(ctx, a, b)
    ccall((:ggml_geglu_split, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_swiglu_split(ctx, a, b)
    ccall((:ggml_swiglu_split, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_geglu_erf_split(ctx, a, b)
    ccall((:ggml_geglu_erf_split, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_geglu_quick_split(ctx, a, b)
    ccall((:ggml_geglu_quick_split, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_swiglu_oai(ctx, a, b, alpha, limit)
    ccall((:ggml_swiglu_oai, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cfloat, Cfloat), ctx, a, b, alpha, limit)
end

function ggml_norm(ctx, a, eps)
    ccall((:ggml_norm, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, eps)
end

function ggml_norm_inplace(ctx, a, eps)
    ccall((:ggml_norm_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, eps)
end

function ggml_rms_norm(ctx, a, eps)
    ccall((:ggml_rms_norm, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, eps)
end

function ggml_rms_norm_inplace(ctx, a, eps)
    ccall((:ggml_rms_norm_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, eps)
end

function ggml_group_norm(ctx, a, n_groups, eps)
    ccall((:ggml_group_norm, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cfloat), ctx, a, n_groups, eps)
end

function ggml_group_norm_inplace(ctx, a, n_groups, eps)
    ccall((:ggml_group_norm_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cfloat), ctx, a, n_groups, eps)
end

function ggml_l2_norm(ctx, a, eps)
    ccall((:ggml_l2_norm, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, eps)
end

function ggml_l2_norm_inplace(ctx, a, eps)
    ccall((:ggml_l2_norm_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, eps)
end

function ggml_rms_norm_back(ctx, a, b, eps)
    ccall((:ggml_rms_norm_back, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cfloat), ctx, a, b, eps)
end

function ggml_mul_mat(ctx, a, b)
    ccall((:ggml_mul_mat, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_mul_mat_set_prec(a, prec)
    ccall((:ggml_mul_mat_set_prec, libllama), Cvoid, (Ptr{ggml_tensor}, ggml_prec), a, prec)
end

function ggml_mul_mat_set_hint(a, hint)
    ccall((:ggml_mul_mat_set_hint, libllama), Cvoid, (Ptr{ggml_tensor}, ggml_op_hint), a, hint)
end

function ggml_mul_mat_id(ctx, as, b, ids)
    ccall((:ggml_mul_mat_id, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, as, b, ids)
end

function ggml_out_prod(ctx, a, b)
    ccall((:ggml_out_prod, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_scale(ctx, a, s)
    ccall((:ggml_scale, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, s)
end

function ggml_scale_inplace(ctx, a, s)
    ccall((:ggml_scale_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, s)
end

function ggml_scale_bias(ctx, a, s, b)
    ccall((:ggml_scale_bias, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat, Cfloat), ctx, a, s, b)
end

function ggml_scale_bias_inplace(ctx, a, s, b)
    ccall((:ggml_scale_bias_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat, Cfloat), ctx, a, s, b)
end

function ggml_set(ctx, a, b, nb1, nb2, nb3, offset)
    ccall((:ggml_set, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t, Csize_t, Csize_t, Csize_t), ctx, a, b, nb1, nb2, nb3, offset)
end

function ggml_set_inplace(ctx, a, b, nb1, nb2, nb3, offset)
    ccall((:ggml_set_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t, Csize_t, Csize_t, Csize_t), ctx, a, b, nb1, nb2, nb3, offset)
end

function ggml_set_1d(ctx, a, b, offset)
    ccall((:ggml_set_1d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t), ctx, a, b, offset)
end

function ggml_set_1d_inplace(ctx, a, b, offset)
    ccall((:ggml_set_1d_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t), ctx, a, b, offset)
end

function ggml_set_2d(ctx, a, b, nb1, offset)
    ccall((:ggml_set_2d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t, Csize_t), ctx, a, b, nb1, offset)
end

function ggml_set_2d_inplace(ctx, a, b, nb1, offset)
    ccall((:ggml_set_2d_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Csize_t, Csize_t), ctx, a, b, nb1, offset)
end

function ggml_cpy(ctx, a, b)
    ccall((:ggml_cpy, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_cast(ctx, a, type)
    ccall((:ggml_cast, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_type), ctx, a, type)
end

function ggml_cont(ctx, a)
    ccall((:ggml_cont, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_cont_1d(ctx, a, ne0)
    ccall((:ggml_cont_1d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64), ctx, a, ne0)
end

function ggml_cont_2d(ctx, a, ne0, ne1)
    ccall((:ggml_cont_2d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64), ctx, a, ne0, ne1)
end

function ggml_cont_3d(ctx, a, ne0, ne1, ne2)
    ccall((:ggml_cont_3d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64), ctx, a, ne0, ne1, ne2)
end

function ggml_cont_4d(ctx, a, ne0, ne1, ne2, ne3)
    ccall((:ggml_cont_4d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64, Int64), ctx, a, ne0, ne1, ne2, ne3)
end

function ggml_reshape(ctx, a, b)
    ccall((:ggml_reshape, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_reshape_1d(ctx, a, ne0)
    ccall((:ggml_reshape_1d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64), ctx, a, ne0)
end

function ggml_reshape_2d(ctx, a, ne0, ne1)
    ccall((:ggml_reshape_2d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64), ctx, a, ne0, ne1)
end

function ggml_reshape_3d(ctx, a, ne0, ne1, ne2)
    ccall((:ggml_reshape_3d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64), ctx, a, ne0, ne1, ne2)
end

function ggml_reshape_4d(ctx, a, ne0, ne1, ne2, ne3)
    ccall((:ggml_reshape_4d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64, Int64), ctx, a, ne0, ne1, ne2, ne3)
end

function ggml_view_1d(ctx, a, ne0, offset)
    ccall((:ggml_view_1d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Csize_t), ctx, a, ne0, offset)
end

function ggml_view_2d(ctx, a, ne0, ne1, nb1, offset)
    ccall((:ggml_view_2d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Csize_t, Csize_t), ctx, a, ne0, ne1, nb1, offset)
end

function ggml_view_3d(ctx, a, ne0, ne1, ne2, nb1, nb2, offset)
    ccall((:ggml_view_3d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64, Csize_t, Csize_t, Csize_t), ctx, a, ne0, ne1, ne2, nb1, nb2, offset)
end

function ggml_view_4d(ctx, a, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset)
    ccall((:ggml_view_4d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64, Int64, Csize_t, Csize_t, Csize_t, Csize_t), ctx, a, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset)
end

function ggml_permute(ctx, a, axis0, axis1, axis2, axis3)
    ccall((:ggml_permute, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint), ctx, a, axis0, axis1, axis2, axis3)
end

function ggml_transpose(ctx, a)
    ccall((:ggml_transpose, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_get_rows(ctx, a, b)
    ccall((:ggml_get_rows, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_get_rows_back(ctx, a, b, c)
    ccall((:ggml_get_rows_back, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b, c)
end

function ggml_set_rows(ctx, a, b, c)
    ccall((:ggml_set_rows, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b, c)
end

function ggml_diag(ctx, a)
    ccall((:ggml_diag, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_diag_mask_inf(ctx, a, n_past)
    ccall((:ggml_diag_mask_inf, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, n_past)
end

function ggml_diag_mask_inf_inplace(ctx, a, n_past)
    ccall((:ggml_diag_mask_inf_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, n_past)
end

function ggml_diag_mask_zero(ctx, a, n_past)
    ccall((:ggml_diag_mask_zero, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, n_past)
end

function ggml_diag_mask_zero_inplace(ctx, a, n_past)
    ccall((:ggml_diag_mask_zero_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, n_past)
end

function ggml_soft_max(ctx, a)
    ccall((:ggml_soft_max, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_soft_max_inplace(ctx, a)
    ccall((:ggml_soft_max_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}), ctx, a)
end

function ggml_soft_max_ext(ctx, a, mask, scale, max_bias)
    ccall((:ggml_soft_max_ext, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cfloat, Cfloat), ctx, a, mask, scale, max_bias)
end

function ggml_soft_max_ext_inplace(ctx, a, mask, scale, max_bias)
    ccall((:ggml_soft_max_ext_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cfloat, Cfloat), ctx, a, mask, scale, max_bias)
end

function ggml_soft_max_add_sinks(a, sinks)
    ccall((:ggml_soft_max_add_sinks, libllama), Cvoid, (Ptr{ggml_tensor}, Ptr{ggml_tensor}), a, sinks)
end

function ggml_soft_max_ext_back(ctx, a, b, scale, max_bias)
    ccall((:ggml_soft_max_ext_back, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cfloat, Cfloat), ctx, a, b, scale, max_bias)
end

function ggml_soft_max_ext_back_inplace(ctx, a, b, scale, max_bias)
    ccall((:ggml_soft_max_ext_back_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cfloat, Cfloat), ctx, a, b, scale, max_bias)
end

function ggml_rope(ctx, a, b, n_dims, mode)
    ccall((:ggml_rope, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint), ctx, a, b, n_dims, mode)
end

function ggml_rope_inplace(ctx, a, b, n_dims, mode)
    ccall((:ggml_rope_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint), ctx, a, b, n_dims, mode)
end

function ggml_rope_ext(ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
    ccall((:ggml_rope_ext, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat), ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
end

function ggml_rope_multi(ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
    ccall((:ggml_rope_multi, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Ptr{Cint}, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat), ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
end

function ggml_rope_ext_inplace(ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
    ccall((:ggml_rope_ext_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat), ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
end

function ggml_rope_multi_inplace(ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
    ccall((:ggml_rope_multi_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Ptr{Cint}, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat), ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
end

function ggml_rope_custom(ctx, a, b, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
    ccall((:ggml_rope_custom, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat), ctx, a, b, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
end

function ggml_rope_custom_inplace(ctx, a, b, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
    ccall((:ggml_rope_custom_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat), ctx, a, b, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
end

function ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, dims)
    ccall((:ggml_rope_yarn_corr_dims, libllama), Cvoid, (Cint, Cint, Cfloat, Cfloat, Cfloat, Ptr{Cfloat}), n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, dims)
end

function ggml_rope_ext_back(ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
    ccall((:ggml_rope_ext_back, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat), ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
end

function ggml_rope_multi_back(ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
    ccall((:ggml_rope_multi_back, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Ptr{Cint}, Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat, Cfloat), ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
end

function ggml_clamp(ctx, a, min, max)
    ccall((:ggml_clamp, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat, Cfloat), ctx, a, min, max)
end

function ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, is_2D, dst_type)
    ccall((:ggml_im2col, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cint, Cint, Bool, ggml_type), ctx, a, b, s0, s1, p0, p1, d0, d1, is_2D, dst_type)
end

function ggml_im2col_back(ctx, a, b, ne, s0, s1, p0, p1, d0, d1, is_2D)
    ccall((:ggml_im2col_back, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{Int64}, Cint, Cint, Cint, Cint, Cint, Cint, Bool), ctx, a, b, ne, s0, s1, p0, p1, d0, d1, is_2D)
end

function ggml_col2im_1d(ctx, a, s0, oc, p0)
    ccall((:ggml_col2im_1d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint), ctx, a, s0, oc, p0)
end

function ggml_conv_1d(ctx, a, b, s0, p0, d0)
    ccall((:ggml_conv_1d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint), ctx, a, b, s0, p0, d0)
end

function ggml_conv_1d_ph(ctx, a, b, s, d)
    ccall((:ggml_conv_1d_ph, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint), ctx, a, b, s, d)
end

function ggml_conv_1d_dw(ctx, a, b, s0, p0, d0)
    ccall((:ggml_conv_1d_dw, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint), ctx, a, b, s0, p0, d0)
end

function ggml_conv_1d_dw_ph(ctx, a, b, s0, d0)
    ccall((:ggml_conv_1d_dw_ph, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint), ctx, a, b, s0, d0)
end

function ggml_conv_transpose_1d(ctx, a, b, s0, p0, d0)
    ccall((:ggml_conv_transpose_1d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint), ctx, a, b, s0, p0, d0)
end

function ggml_conv_2d(ctx, a, b, s0, s1, p0, p1, d0, d1)
    ccall((:ggml_conv_2d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cint, Cint), ctx, a, b, s0, s1, p0, p1, d0, d1)
end

function ggml_im2col_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, dst_type)
    ccall((:ggml_im2col_3d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Int64, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, ggml_type), ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, dst_type)
end

function ggml_conv_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2)
    ccall((:ggml_conv_3d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Int64, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint), ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2)
end

function ggml_conv_2d_sk_p0(ctx, a, b)
    ccall((:ggml_conv_2d_sk_p0, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_conv_2d_s1_ph(ctx, a, b)
    ccall((:ggml_conv_2d_s1_ph, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_conv_2d_dw(ctx, a, b, s0, s1, p0, p1, d0, d1)
    ccall((:ggml_conv_2d_dw, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cint, Cint), ctx, a, b, s0, s1, p0, p1, d0, d1)
end

function ggml_conv_2d_dw_direct(ctx, a, b, stride0, stride1, pad0, pad1, dilation0, dilation1)
    ccall((:ggml_conv_2d_dw_direct, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cint, Cint), ctx, a, b, stride0, stride1, pad0, pad1, dilation0, dilation1)
end

function ggml_conv_transpose_2d_p0(ctx, a, b, stride)
    ccall((:ggml_conv_transpose_2d_p0, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint), ctx, a, b, stride)
end

function ggml_conv_2d_direct(ctx, a, b, s0, s1, p0, p1, d0, d1)
    ccall((:ggml_conv_2d_direct, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cint, Cint), ctx, a, b, s0, s1, p0, p1, d0, d1)
end

function ggml_conv_3d_direct(ctx, a, b, s0, s1, s2, p0, p1, p2, d0, d1, d2, n_channels, n_batch, n_channels_out)
    ccall((:ggml_conv_3d_direct, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint), ctx, a, b, s0, s1, s2, p0, p1, p2, d0, d1, d2, n_channels, n_batch, n_channels_out)
end

@cenum ggml_op_pool::UInt32 begin
    GGML_OP_POOL_MAX = 0
    GGML_OP_POOL_AVG = 1
    GGML_OP_POOL_COUNT = 2
end

function ggml_pool_1d(ctx, a, op, k0, s0, p0)
    ccall((:ggml_pool_1d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_op_pool, Cint, Cint, Cint), ctx, a, op, k0, s0, p0)
end

function ggml_pool_2d(ctx, a, op, k0, k1, s0, s1, p0, p1)
    ccall((:ggml_pool_2d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_op_pool, Cint, Cint, Cint, Cint, Cfloat, Cfloat), ctx, a, op, k0, k1, s0, s1, p0, p1)
end

function ggml_pool_2d_back(ctx, a, af, op, k0, k1, s0, s1, p0, p1)
    ccall((:ggml_pool_2d_back, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_op_pool, Cint, Cint, Cint, Cint, Cfloat, Cfloat), ctx, a, af, op, k0, k1, s0, s1, p0, p1)
end

@cenum ggml_scale_mode::UInt32 begin
    GGML_SCALE_MODE_NEAREST = 0
    GGML_SCALE_MODE_BILINEAR = 1
    GGML_SCALE_MODE_BICUBIC = 2
    GGML_SCALE_MODE_COUNT = 3
end

@cenum ggml_scale_flag::UInt32 begin
    GGML_SCALE_FLAG_ALIGN_CORNERS = 256
    GGML_SCALE_FLAG_ANTIALIAS = 512
end

function ggml_upscale(ctx, a, scale_factor, mode)
    ccall((:ggml_upscale, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, ggml_scale_mode), ctx, a, scale_factor, mode)
end

function ggml_upscale_ext(ctx, a, ne0, ne1, ne2, ne3, mode)
    ccall((:ggml_upscale_ext, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, ggml_scale_mode), ctx, a, ne0, ne1, ne2, ne3, mode)
end

function ggml_interpolate(ctx, a, ne0, ne1, ne2, ne3, mode)
    ccall((:ggml_interpolate, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Int64, Int64, Int64, Int64, UInt32), ctx, a, ne0, ne1, ne2, ne3, mode)
end

function ggml_pad(ctx, a, p0, p1, p2, p3)
    ccall((:ggml_pad, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint), ctx, a, p0, p1, p2, p3)
end

function ggml_pad_circular(ctx, a, p0, p1, p2, p3)
    ccall((:ggml_pad_circular, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint), ctx, a, p0, p1, p2, p3)
end

function ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3)
    ccall((:ggml_pad_ext, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint), ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3)
end

function ggml_pad_ext_circular(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3)
    ccall((:ggml_pad_ext_circular, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint), ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3)
end

function ggml_pad_reflect_1d(ctx, a, p0, p1)
    ccall((:ggml_pad_reflect_1d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint), ctx, a, p0, p1)
end

function ggml_roll(ctx, a, shift0, shift1, shift2, shift3)
    ccall((:ggml_roll, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint, Cint), ctx, a, shift0, shift1, shift2, shift3)
end

function ggml_tri(ctx, a, type)
    ccall((:ggml_tri, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_tri_type), ctx, a, type)
end

function ggml_fill(ctx, a, c)
    ccall((:ggml_fill, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, c)
end

function ggml_fill_inplace(ctx, a, c)
    ccall((:ggml_fill_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cfloat), ctx, a, c)
end

function ggml_timestep_embedding(ctx, timesteps, dim, max_period)
    ccall((:ggml_timestep_embedding, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint), ctx, timesteps, dim, max_period)
end

@cenum ggml_sort_order::UInt32 begin
    GGML_SORT_ORDER_ASC = 0
    GGML_SORT_ORDER_DESC = 1
end

function ggml_argsort(ctx, a, order)
    ccall((:ggml_argsort, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_sort_order), ctx, a, order)
end

function ggml_argsort_top_k(ctx, a, k)
    ccall((:ggml_argsort_top_k, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, k)
end

function ggml_top_k(ctx, a, k)
    ccall((:ggml_top_k, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, k)
end

function ggml_arange(ctx, start, stop, step)
    ccall((:ggml_arange, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Cfloat, Cfloat, Cfloat), ctx, start, stop, step)
end

function ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias, logit_softcap)
    ccall((:ggml_flash_attn_ext, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cfloat, Cfloat, Cfloat), ctx, q, k, v, mask, scale, max_bias, logit_softcap)
end

function ggml_flash_attn_ext_set_prec(a, prec)
    ccall((:ggml_flash_attn_ext_set_prec, libllama), Cvoid, (Ptr{ggml_tensor}, ggml_prec), a, prec)
end

function ggml_flash_attn_ext_get_prec(a)
    ccall((:ggml_flash_attn_ext_get_prec, libllama), ggml_prec, (Ptr{ggml_tensor},), a)
end

function ggml_flash_attn_ext_add_sinks(a, sinks)
    ccall((:ggml_flash_attn_ext_add_sinks, libllama), Cvoid, (Ptr{ggml_tensor}, Ptr{ggml_tensor}), a, sinks)
end

function ggml_flash_attn_back(ctx, q, k, v, d, masked)
    ccall((:ggml_flash_attn_back, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Bool), ctx, q, k, v, d, masked)
end

function ggml_ssm_conv(ctx, sx, c)
    ccall((:ggml_ssm_conv, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, sx, c)
end

function ggml_ssm_scan(ctx, s, x, dt, A, B, C, ids)
    ccall((:ggml_ssm_scan, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, s, x, dt, A, B, C, ids)
end

function ggml_win_part(ctx, a, w)
    ccall((:ggml_win_part, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint), ctx, a, w)
end

function ggml_win_unpart(ctx, a, w0, h0, w)
    ccall((:ggml_win_unpart, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint, Cint), ctx, a, w0, h0, w)
end

function ggml_unary(ctx, a, op)
    ccall((:ggml_unary, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_unary_op), ctx, a, op)
end

function ggml_unary_inplace(ctx, a, op)
    ccall((:ggml_unary_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_unary_op), ctx, a, op)
end

function ggml_get_rel_pos(ctx, a, qh, kh)
    ccall((:ggml_get_rel_pos, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Cint, Cint), ctx, a, qh, kh)
end

function ggml_add_rel_pos(ctx, a, pw, ph)
    ccall((:ggml_add_rel_pos, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, pw, ph)
end

function ggml_add_rel_pos_inplace(ctx, a, pw, ph)
    ccall((:ggml_add_rel_pos_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, pw, ph)
end

function ggml_rwkv_wkv6(ctx, k, v, r, tf, td, state)
    ccall((:ggml_rwkv_wkv6, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, k, v, r, tf, td, state)
end

function ggml_gated_linear_attn(ctx, k, v, q, g, state, scale)
    ccall((:ggml_gated_linear_attn, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cfloat), ctx, k, v, q, g, state, scale)
end

function ggml_rwkv_wkv7(ctx, r, w, k, v, a, b, state)
    ccall((:ggml_rwkv_wkv7, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, r, w, k, v, a, b, state)
end

function ggml_solve_tri(ctx, a, b, left, lower, uni)
    ccall((:ggml_solve_tri, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Bool, Bool, Bool), ctx, a, b, left, lower, uni)
end

function ggml_gated_delta_net(ctx, q, k, v, g, beta, state, K)
    ccall((:ggml_gated_delta_net, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Int64), ctx, q, k, v, g, beta, state, K)
end

function ggml_lightning_indexer(ctx, q, k, weights, mask)
    ccall((:ggml_lightning_indexer, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, q, k, weights, mask)
end

function ggml_dsv4_hc_comb(ctx, mixes, scale, base, eps, n_iter)
    ccall((:ggml_dsv4_hc_comb, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Cfloat, Int32), ctx, mixes, scale, base, eps, n_iter)
end

function ggml_dsv4_hc_pre(ctx, x, weights)
    ccall((:ggml_dsv4_hc_pre, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, x, weights)
end

function ggml_dsv4_hc_post(ctx, x, residual, post, comb)
    ccall((:ggml_dsv4_hc_post, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, x, residual, post, comb)
end

# typedef void ( * ggml_custom1_op_t ) ( struct ggml_tensor * dst , const struct ggml_tensor * a , int ith , int nth , void * userdata )
const ggml_custom1_op_t = Ptr{Cvoid}

# typedef void ( * ggml_custom2_op_t ) ( struct ggml_tensor * dst , const struct ggml_tensor * a , const struct ggml_tensor * b , int ith , int nth , void * userdata )
const ggml_custom2_op_t = Ptr{Cvoid}

# typedef void ( * ggml_custom3_op_t ) ( struct ggml_tensor * dst , const struct ggml_tensor * a , const struct ggml_tensor * b , const struct ggml_tensor * c , int ith , int nth , void * userdata )
const ggml_custom3_op_t = Ptr{Cvoid}

function ggml_map_custom1(ctx, a, fun, n_tasks, userdata)
    ccall((:ggml_map_custom1, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_custom1_op_t, Cint, Ptr{Cvoid}), ctx, a, fun, n_tasks, userdata)
end

function ggml_map_custom1_inplace(ctx, a, fun, n_tasks, userdata)
    ccall((:ggml_map_custom1_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, ggml_custom1_op_t, Cint, Ptr{Cvoid}), ctx, a, fun, n_tasks, userdata)
end

function ggml_map_custom2(ctx, a, b, fun, n_tasks, userdata)
    ccall((:ggml_map_custom2, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_custom2_op_t, Cint, Ptr{Cvoid}), ctx, a, b, fun, n_tasks, userdata)
end

function ggml_map_custom2_inplace(ctx, a, b, fun, n_tasks, userdata)
    ccall((:ggml_map_custom2_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_custom2_op_t, Cint, Ptr{Cvoid}), ctx, a, b, fun, n_tasks, userdata)
end

function ggml_map_custom3(ctx, a, b, c, fun, n_tasks, userdata)
    ccall((:ggml_map_custom3, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_custom3_op_t, Cint, Ptr{Cvoid}), ctx, a, b, c, fun, n_tasks, userdata)
end

function ggml_map_custom3_inplace(ctx, a, b, c, fun, n_tasks, userdata)
    ccall((:ggml_map_custom3_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_custom3_op_t, Cint, Ptr{Cvoid}), ctx, a, b, c, fun, n_tasks, userdata)
end

# typedef void ( * ggml_custom_op_t ) ( struct ggml_tensor * dst , int ith , int nth , void * userdata )
const ggml_custom_op_t = Ptr{Cvoid}

function ggml_custom_4d(ctx, type, ne0, ne1, ne2, ne3, args, n_args, fun, n_tasks, userdata)
    ccall((:ggml_custom_4d, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, ggml_type, Int64, Int64, Int64, Int64, Ptr{Ptr{ggml_tensor}}, Cint, ggml_custom_op_t, Cint, Ptr{Cvoid}), ctx, type, ne0, ne1, ne2, ne3, args, n_args, fun, n_tasks, userdata)
end

function ggml_custom_inplace(ctx, a, args, n_args, fun, n_tasks, userdata)
    ccall((:ggml_custom_inplace, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{Ptr{ggml_tensor}}, Cint, ggml_custom_op_t, Cint, Ptr{Cvoid}), ctx, a, args, n_args, fun, n_tasks, userdata)
end

function ggml_cross_entropy_loss(ctx, a, b)
    ccall((:ggml_cross_entropy_loss, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b)
end

function ggml_cross_entropy_loss_back(ctx, a, b, c)
    ccall((:ggml_cross_entropy_loss_back, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, b, c)
end

function ggml_opt_step_adamw(ctx, a, grad, m, v, adamw_params)
    ccall((:ggml_opt_step_adamw, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, grad, m, v, adamw_params)
end

function ggml_opt_step_sgd(ctx, a, grad, sgd_params)
    ccall((:ggml_opt_step_sgd, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), ctx, a, grad, sgd_params)
end

mutable struct ggml_cgraph end

function ggml_build_forward_select(cgraph, tensors, n_tensors, idx)
    ccall((:ggml_build_forward_select, libllama), Ptr{ggml_tensor}, (Ptr{ggml_cgraph}, Ptr{Ptr{ggml_tensor}}, Cint, Cint), cgraph, tensors, n_tensors, idx)
end

function ggml_build_forward_expand(cgraph, tensor)
    ccall((:ggml_build_forward_expand, libllama), Cvoid, (Ptr{ggml_cgraph}, Ptr{ggml_tensor}), cgraph, tensor)
end

function ggml_build_forward_order(cgraph, tensor)
    ccall((:ggml_build_forward_order, libllama), Cvoid, (Ptr{ggml_cgraph}, Ptr{ggml_tensor}), cgraph, tensor)
end

function ggml_build_backward_expand(ctx, cgraph, grad_accs)
    ccall((:ggml_build_backward_expand, libllama), Cvoid, (Ptr{ggml_context}, Ptr{ggml_cgraph}, Ptr{Ptr{ggml_tensor}}), ctx, cgraph, grad_accs)
end

function ggml_new_graph(ctx)
    ccall((:ggml_new_graph, libllama), Ptr{ggml_cgraph}, (Ptr{ggml_context},), ctx)
end

function ggml_new_graph_custom(ctx, size, grads)
    ccall((:ggml_new_graph_custom, libllama), Ptr{ggml_cgraph}, (Ptr{ggml_context}, Csize_t, Bool), ctx, size, grads)
end

function ggml_graph_dup(ctx, cgraph, force_grads)
    ccall((:ggml_graph_dup, libllama), Ptr{ggml_cgraph}, (Ptr{ggml_context}, Ptr{ggml_cgraph}, Bool), ctx, cgraph, force_grads)
end

function ggml_graph_cpy(src, dst)
    ccall((:ggml_graph_cpy, libllama), Cvoid, (Ptr{ggml_cgraph}, Ptr{ggml_cgraph}), src, dst)
end

function ggml_graph_reset(cgraph)
    ccall((:ggml_graph_reset, libllama), Cvoid, (Ptr{ggml_cgraph},), cgraph)
end

function ggml_graph_clear(cgraph)
    ccall((:ggml_graph_clear, libllama), Cvoid, (Ptr{ggml_cgraph},), cgraph)
end

function ggml_graph_size(cgraph)
    ccall((:ggml_graph_size, libllama), Cint, (Ptr{ggml_cgraph},), cgraph)
end

function ggml_graph_node(cgraph, i)
    ccall((:ggml_graph_node, libllama), Ptr{ggml_tensor}, (Ptr{ggml_cgraph}, Cint), cgraph, i)
end

function ggml_graph_nodes(cgraph)
    ccall((:ggml_graph_nodes, libllama), Ptr{Ptr{ggml_tensor}}, (Ptr{ggml_cgraph},), cgraph)
end

function ggml_graph_n_nodes(cgraph)
    ccall((:ggml_graph_n_nodes, libllama), Cint, (Ptr{ggml_cgraph},), cgraph)
end

function ggml_graph_add_node(cgraph, tensor)
    ccall((:ggml_graph_add_node, libllama), Cvoid, (Ptr{ggml_cgraph}, Ptr{ggml_tensor}), cgraph, tensor)
end

function ggml_graph_overhead()
    ccall((:ggml_graph_overhead, libllama), Csize_t, ())
end

function ggml_graph_overhead_custom(size, grads)
    ccall((:ggml_graph_overhead_custom, libllama), Csize_t, (Csize_t, Bool), size, grads)
end

function ggml_graph_get_tensor(cgraph, name)
    ccall((:ggml_graph_get_tensor, libllama), Ptr{ggml_tensor}, (Ptr{ggml_cgraph}, Ptr{Cchar}), cgraph, name)
end

function ggml_graph_get_grad(cgraph, node)
    ccall((:ggml_graph_get_grad, libllama), Ptr{ggml_tensor}, (Ptr{ggml_cgraph}, Ptr{ggml_tensor}), cgraph, node)
end

function ggml_graph_get_grad_acc(cgraph, node)
    ccall((:ggml_graph_get_grad_acc, libllama), Ptr{ggml_tensor}, (Ptr{ggml_cgraph}, Ptr{ggml_tensor}), cgraph, node)
end

function ggml_graph_print(cgraph)
    ccall((:ggml_graph_print, libllama), Cvoid, (Ptr{ggml_cgraph},), cgraph)
end

function ggml_graph_dump_dot(gb, cgraph, filename)
    ccall((:ggml_graph_dump_dot, libllama), Cvoid, (Ptr{ggml_cgraph}, Ptr{ggml_cgraph}, Ptr{Cchar}), gb, cgraph, filename)
end

# typedef void ( * ggml_log_callback ) ( enum ggml_log_level level , const char * text , void * user_data )
const ggml_log_callback = Ptr{Cvoid}

function ggml_log_get(log_callback, user_data)
    ccall((:ggml_log_get, libllama), Cvoid, (Ptr{ggml_log_callback}, Ptr{Ptr{Cvoid}}), log_callback, user_data)
end

function ggml_log_set(log_callback, user_data)
    ccall((:ggml_log_set, libllama), Cvoid, (ggml_log_callback, Ptr{Cvoid}), log_callback, user_data)
end

function ggml_set_zero(tensor)
    ccall((:ggml_set_zero, libllama), Ptr{ggml_tensor}, (Ptr{ggml_tensor},), tensor)
end

function ggml_quantize_init(type)
    ccall((:ggml_quantize_init, libllama), Cvoid, (ggml_type,), type)
end

function ggml_quantize_free()
    ccall((:ggml_quantize_free, libllama), Cvoid, ())
end

function ggml_quantize_requires_imatrix(type)
    ccall((:ggml_quantize_requires_imatrix, libllama), Bool, (ggml_type,), type)
end

function ggml_quantize_chunk(type, src, dst, start, nrows, n_per_row, imatrix)
    ccall((:ggml_quantize_chunk, libllama), Csize_t, (ggml_type, Ptr{Cfloat}, Ptr{Cvoid}, Int64, Int64, Int64, Ptr{Cfloat}), type, src, dst, start, nrows, n_per_row, imatrix)
end

# typedef void ( * ggml_to_float_t ) ( const void * GGML_RESTRICT x , float * GGML_RESTRICT y , int64_t k )
const ggml_to_float_t = Ptr{Cvoid}

# typedef void ( * ggml_from_float_t ) ( const float * GGML_RESTRICT x , void * GGML_RESTRICT y , int64_t k )
const ggml_from_float_t = Ptr{Cvoid}

struct ggml_type_traits
    type_name::Ptr{Cchar}
    blck_size::Int64
    blck_size_interleave::Int64
    type_size::Csize_t
    is_quantized::Bool
    to_float::ggml_to_float_t
    from_float_ref::ggml_from_float_t
end

function ggml_get_type_traits(type)
    ccall((:ggml_get_type_traits, libllama), Ptr{ggml_type_traits}, (ggml_type,), type)
end

@cenum ggml_sched_priority::Int32 begin
    GGML_SCHED_PRIO_LOW = -1
    GGML_SCHED_PRIO_NORMAL = 0
    GGML_SCHED_PRIO_MEDIUM = 1
    GGML_SCHED_PRIO_HIGH = 2
    GGML_SCHED_PRIO_REALTIME = 3
end

struct ggml_threadpool_params
    cpumask::NTuple{512, Bool}
    n_threads::Cint
    prio::ggml_sched_priority
    poll::UInt32
    strict_cpu::Bool
    paused::Bool
end

mutable struct ggml_threadpool end

const ggml_threadpool_t = Ptr{ggml_threadpool}

function ggml_threadpool_params_default(n_threads)
    ccall((:ggml_threadpool_params_default, libllama), ggml_threadpool_params, (Cint,), n_threads)
end

function ggml_threadpool_params_init(p, n_threads)
    ccall((:ggml_threadpool_params_init, libllama), Cvoid, (Ptr{ggml_threadpool_params}, Cint), p, n_threads)
end

function ggml_threadpool_params_match(p0, p1)
    ccall((:ggml_threadpool_params_match, libllama), Bool, (Ptr{ggml_threadpool_params}, Ptr{ggml_threadpool_params}), p0, p1)
end

mutable struct ggml_backend_buffer_type end

const ggml_backend_buffer_type_t = Ptr{ggml_backend_buffer_type}

const ggml_backend_buffer_t = Ptr{ggml_backend_buffer}

mutable struct ggml_backend end

const ggml_backend_t = Ptr{ggml_backend}

struct ggml_tallocr
    buffer::ggml_backend_buffer_t
    base::Ptr{Cvoid}
    alignment::Csize_t
    offset::Csize_t
end

function ggml_tallocr_new(buffer)
    ccall((:ggml_tallocr_new, libllama), ggml_tallocr, (ggml_backend_buffer_t,), buffer)
end

function ggml_tallocr_alloc(talloc, tensor)
    ccall((:ggml_tallocr_alloc, libllama), ggml_status, (Ptr{ggml_tallocr}, Ptr{ggml_tensor}), talloc, tensor)
end

mutable struct ggml_gallocr end

const ggml_gallocr_t = Ptr{ggml_gallocr}

function ggml_gallocr_new(buft)
    ccall((:ggml_gallocr_new, libllama), ggml_gallocr_t, (ggml_backend_buffer_type_t,), buft)
end

function ggml_gallocr_new_n(bufts, n_bufs)
    ccall((:ggml_gallocr_new_n, libllama), ggml_gallocr_t, (Ptr{ggml_backend_buffer_type_t}, Cint), bufts, n_bufs)
end

function ggml_gallocr_free(galloc)
    ccall((:ggml_gallocr_free, libllama), Cvoid, (ggml_gallocr_t,), galloc)
end

function ggml_gallocr_reserve(galloc, graph)
    ccall((:ggml_gallocr_reserve, libllama), Bool, (ggml_gallocr_t, Ptr{ggml_cgraph}), galloc, graph)
end

function ggml_gallocr_reserve_n_size(galloc, graph, node_buffer_ids, leaf_buffer_ids, sizes)
    ccall((:ggml_gallocr_reserve_n_size, libllama), Cvoid, (ggml_gallocr_t, Ptr{ggml_cgraph}, Ptr{Cint}, Ptr{Cint}, Ptr{Csize_t}), galloc, graph, node_buffer_ids, leaf_buffer_ids, sizes)
end

function ggml_gallocr_reserve_n(galloc, graph, node_buffer_ids, leaf_buffer_ids)
    ccall((:ggml_gallocr_reserve_n, libllama), Bool, (ggml_gallocr_t, Ptr{ggml_cgraph}, Ptr{Cint}, Ptr{Cint}), galloc, graph, node_buffer_ids, leaf_buffer_ids)
end

function ggml_gallocr_alloc_graph(galloc, graph)
    ccall((:ggml_gallocr_alloc_graph, libllama), Bool, (ggml_gallocr_t, Ptr{ggml_cgraph}), galloc, graph)
end

function ggml_gallocr_get_buffer_size(galloc, buffer_id)
    ccall((:ggml_gallocr_get_buffer_size, libllama), Csize_t, (ggml_gallocr_t, Cint), galloc, buffer_id)
end

function ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft)
    ccall((:ggml_backend_alloc_ctx_tensors_from_buft_size, libllama), Csize_t, (Ptr{ggml_context}, ggml_backend_buffer_type_t), ctx, buft)
end

function ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)
    ccall((:ggml_backend_alloc_ctx_tensors_from_buft, libllama), Ptr{ggml_backend_buffer}, (Ptr{ggml_context}, ggml_backend_buffer_type_t), ctx, buft)
end

function ggml_backend_alloc_ctx_tensors(ctx, backend)
    ccall((:ggml_backend_alloc_ctx_tensors, libllama), Ptr{ggml_backend_buffer}, (Ptr{ggml_context}, ggml_backend_t), ctx, backend)
end

mutable struct ggml_backend_event end

const ggml_backend_event_t = Ptr{ggml_backend_event}

const ggml_backend_graph_plan_t = Ptr{Cvoid}

mutable struct ggml_backend_reg end

const ggml_backend_reg_t = Ptr{ggml_backend_reg}

mutable struct ggml_backend_device end

const ggml_backend_dev_t = Ptr{ggml_backend_device}

function ggml_backend_buft_name(buft)
    ccall((:ggml_backend_buft_name, libllama), Ptr{Cchar}, (ggml_backend_buffer_type_t,), buft)
end

function ggml_backend_buft_alloc_buffer(buft, size)
    ccall((:ggml_backend_buft_alloc_buffer, libllama), ggml_backend_buffer_t, (ggml_backend_buffer_type_t, Csize_t), buft, size)
end

function ggml_backend_buft_get_alignment(buft)
    ccall((:ggml_backend_buft_get_alignment, libllama), Csize_t, (ggml_backend_buffer_type_t,), buft)
end

function ggml_backend_buft_get_max_size(buft)
    ccall((:ggml_backend_buft_get_max_size, libllama), Csize_t, (ggml_backend_buffer_type_t,), buft)
end

function ggml_backend_buft_get_alloc_size(buft, tensor)
    ccall((:ggml_backend_buft_get_alloc_size, libllama), Csize_t, (ggml_backend_buffer_type_t, Ptr{ggml_tensor}), buft, tensor)
end

function ggml_backend_buft_is_host(buft)
    ccall((:ggml_backend_buft_is_host, libllama), Bool, (ggml_backend_buffer_type_t,), buft)
end

function ggml_backend_buft_get_device(buft)
    ccall((:ggml_backend_buft_get_device, libllama), ggml_backend_dev_t, (ggml_backend_buffer_type_t,), buft)
end

@cenum ggml_backend_buffer_usage::UInt32 begin
    GGML_BACKEND_BUFFER_USAGE_ANY = 0
    GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1
    GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2
end

function ggml_backend_buffer_name(buffer)
    ccall((:ggml_backend_buffer_name, libllama), Ptr{Cchar}, (ggml_backend_buffer_t,), buffer)
end

function ggml_backend_buffer_free(buffer)
    ccall((:ggml_backend_buffer_free, libllama), Cvoid, (ggml_backend_buffer_t,), buffer)
end

function ggml_backend_buffer_get_base(buffer)
    ccall((:ggml_backend_buffer_get_base, libllama), Ptr{Cvoid}, (ggml_backend_buffer_t,), buffer)
end

function ggml_backend_buffer_get_size(buffer)
    ccall((:ggml_backend_buffer_get_size, libllama), Csize_t, (ggml_backend_buffer_t,), buffer)
end

function ggml_backend_buffer_init_tensor(buffer, tensor)
    ccall((:ggml_backend_buffer_init_tensor, libllama), ggml_status, (ggml_backend_buffer_t, Ptr{ggml_tensor}), buffer, tensor)
end

function ggml_backend_buffer_get_alignment(buffer)
    ccall((:ggml_backend_buffer_get_alignment, libllama), Csize_t, (ggml_backend_buffer_t,), buffer)
end

function ggml_backend_buffer_get_max_size(buffer)
    ccall((:ggml_backend_buffer_get_max_size, libllama), Csize_t, (ggml_backend_buffer_t,), buffer)
end

function ggml_backend_buffer_get_alloc_size(buffer, tensor)
    ccall((:ggml_backend_buffer_get_alloc_size, libllama), Csize_t, (ggml_backend_buffer_t, Ptr{ggml_tensor}), buffer, tensor)
end

function ggml_backend_buffer_clear(buffer, value)
    ccall((:ggml_backend_buffer_clear, libllama), Cvoid, (ggml_backend_buffer_t, UInt8), buffer, value)
end

function ggml_backend_buffer_is_host(buffer)
    ccall((:ggml_backend_buffer_is_host, libllama), Bool, (ggml_backend_buffer_t,), buffer)
end

function ggml_backend_buffer_set_usage(buffer, usage)
    ccall((:ggml_backend_buffer_set_usage, libllama), Cvoid, (ggml_backend_buffer_t, ggml_backend_buffer_usage), buffer, usage)
end

function ggml_backend_buffer_get_usage(buffer)
    ccall((:ggml_backend_buffer_get_usage, libllama), ggml_backend_buffer_usage, (ggml_backend_buffer_t,), buffer)
end

function ggml_backend_buffer_get_type(buffer)
    ccall((:ggml_backend_buffer_get_type, libllama), ggml_backend_buffer_type_t, (ggml_backend_buffer_t,), buffer)
end

function ggml_backend_buffer_reset(buffer)
    ccall((:ggml_backend_buffer_reset, libllama), Cvoid, (ggml_backend_buffer_t,), buffer)
end

function ggml_backend_tensor_copy(src, dst)
    ccall((:ggml_backend_tensor_copy, libllama), Cvoid, (Ptr{ggml_tensor}, Ptr{ggml_tensor}), src, dst)
end

function ggml_backend_guid(backend)
    ccall((:ggml_backend_guid, libllama), ggml_guid_t, (ggml_backend_t,), backend)
end

function ggml_backend_name(backend)
    ccall((:ggml_backend_name, libllama), Ptr{Cchar}, (ggml_backend_t,), backend)
end

function ggml_backend_free(backend)
    ccall((:ggml_backend_free, libllama), Cvoid, (ggml_backend_t,), backend)
end

function ggml_backend_get_default_buffer_type(backend)
    ccall((:ggml_backend_get_default_buffer_type, libllama), ggml_backend_buffer_type_t, (ggml_backend_t,), backend)
end

function ggml_backend_alloc_buffer(backend, size)
    ccall((:ggml_backend_alloc_buffer, libllama), ggml_backend_buffer_t, (ggml_backend_t, Csize_t), backend, size)
end

function ggml_backend_get_alignment(backend)
    ccall((:ggml_backend_get_alignment, libllama), Csize_t, (ggml_backend_t,), backend)
end

function ggml_backend_get_max_size(backend)
    ccall((:ggml_backend_get_max_size, libllama), Csize_t, (ggml_backend_t,), backend)
end

function ggml_backend_tensor_set_async(backend, tensor, data, offset, size)
    ccall((:ggml_backend_tensor_set_async, libllama), Cvoid, (ggml_backend_t, Ptr{ggml_tensor}, Ptr{Cvoid}, Csize_t, Csize_t), backend, tensor, data, offset, size)
end

function ggml_backend_tensor_get_async(backend, tensor, data, offset, size)
    ccall((:ggml_backend_tensor_get_async, libllama), Cvoid, (ggml_backend_t, Ptr{ggml_tensor}, Ptr{Cvoid}, Csize_t, Csize_t), backend, tensor, data, offset, size)
end

function ggml_backend_tensor_set_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data)
    ccall((:ggml_backend_tensor_set_2d_async, libllama), Cvoid, (ggml_backend_t, Ptr{ggml_tensor}, Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Csize_t, Csize_t), backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data)
end

function ggml_backend_tensor_get_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data)
    ccall((:ggml_backend_tensor_get_2d_async, libllama), Cvoid, (ggml_backend_t, Ptr{ggml_tensor}, Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Csize_t, Csize_t), backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data)
end

function ggml_backend_tensor_set(tensor, data, offset, size)
    ccall((:ggml_backend_tensor_set, libllama), Cvoid, (Ptr{ggml_tensor}, Ptr{Cvoid}, Csize_t, Csize_t), tensor, data, offset, size)
end

function ggml_backend_tensor_get(tensor, data, offset, size)
    ccall((:ggml_backend_tensor_get, libllama), Cvoid, (Ptr{ggml_tensor}, Ptr{Cvoid}, Csize_t, Csize_t), tensor, data, offset, size)
end

function ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data)
    ccall((:ggml_backend_tensor_set_2d, libllama), Cvoid, (Ptr{ggml_tensor}, Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Csize_t, Csize_t), tensor, data, offset, size, n_copies, stride_tensor, stride_data)
end

function ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data)
    ccall((:ggml_backend_tensor_get_2d, libllama), Cvoid, (Ptr{ggml_tensor}, Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Csize_t, Csize_t), tensor, data, offset, size, n_copies, stride_tensor, stride_data)
end

function ggml_backend_tensor_memset(tensor, value, offset, size)
    ccall((:ggml_backend_tensor_memset, libllama), Cvoid, (Ptr{ggml_tensor}, UInt8, Csize_t, Csize_t), tensor, value, offset, size)
end

function ggml_backend_synchronize(backend)
    ccall((:ggml_backend_synchronize, libllama), Cvoid, (ggml_backend_t,), backend)
end

function ggml_backend_graph_plan_create(backend, cgraph)
    ccall((:ggml_backend_graph_plan_create, libllama), ggml_backend_graph_plan_t, (ggml_backend_t, Ptr{ggml_cgraph}), backend, cgraph)
end

function ggml_backend_graph_plan_free(backend, plan)
    ccall((:ggml_backend_graph_plan_free, libllama), Cvoid, (ggml_backend_t, ggml_backend_graph_plan_t), backend, plan)
end

function ggml_backend_graph_plan_compute(backend, plan)
    ccall((:ggml_backend_graph_plan_compute, libllama), ggml_status, (ggml_backend_t, ggml_backend_graph_plan_t), backend, plan)
end

function ggml_backend_graph_compute(backend, cgraph)
    ccall((:ggml_backend_graph_compute, libllama), ggml_status, (ggml_backend_t, Ptr{ggml_cgraph}), backend, cgraph)
end

function ggml_backend_graph_compute_async(backend, cgraph)
    ccall((:ggml_backend_graph_compute_async, libllama), ggml_status, (ggml_backend_t, Ptr{ggml_cgraph}), backend, cgraph)
end

function ggml_backend_supports_op(backend, op)
    ccall((:ggml_backend_supports_op, libllama), Bool, (ggml_backend_t, Ptr{ggml_tensor}), backend, op)
end

function ggml_backend_supports_buft(backend, buft)
    ccall((:ggml_backend_supports_buft, libllama), Bool, (ggml_backend_t, ggml_backend_buffer_type_t), backend, buft)
end

function ggml_backend_offload_op(backend, op)
    ccall((:ggml_backend_offload_op, libllama), Bool, (ggml_backend_t, Ptr{ggml_tensor}), backend, op)
end

function ggml_backend_tensor_copy_async(backend_src, backend_dst, src, dst)
    ccall((:ggml_backend_tensor_copy_async, libllama), Cvoid, (ggml_backend_t, ggml_backend_t, Ptr{ggml_tensor}, Ptr{ggml_tensor}), backend_src, backend_dst, src, dst)
end

function ggml_backend_get_device(backend)
    ccall((:ggml_backend_get_device, libllama), ggml_backend_dev_t, (ggml_backend_t,), backend)
end

function ggml_backend_event_new(device)
    ccall((:ggml_backend_event_new, libllama), ggml_backend_event_t, (ggml_backend_dev_t,), device)
end

function ggml_backend_event_free(event)
    ccall((:ggml_backend_event_free, libllama), Cvoid, (ggml_backend_event_t,), event)
end

function ggml_backend_event_record(event, backend)
    ccall((:ggml_backend_event_record, libllama), Cvoid, (ggml_backend_event_t, ggml_backend_t), event, backend)
end

function ggml_backend_event_synchronize(event)
    ccall((:ggml_backend_event_synchronize, libllama), Cvoid, (ggml_backend_event_t,), event)
end

function ggml_backend_event_wait(backend, event)
    ccall((:ggml_backend_event_wait, libllama), Cvoid, (ggml_backend_t, ggml_backend_event_t), backend, event)
end

@cenum ggml_backend_dev_type::UInt32 begin
    GGML_BACKEND_DEVICE_TYPE_CPU = 0
    GGML_BACKEND_DEVICE_TYPE_GPU = 1
    GGML_BACKEND_DEVICE_TYPE_IGPU = 2
    GGML_BACKEND_DEVICE_TYPE_ACCEL = 3
    GGML_BACKEND_DEVICE_TYPE_META = 4
end

struct ggml_backend_dev_caps
    async::Bool
    host_buffer::Bool
    buffer_from_host_ptr::Bool
    events::Bool
    mmap_support::Bool
end

struct ggml_backend_dev_props
    name::Ptr{Cchar}
    description::Ptr{Cchar}
    memory_free::Csize_t
    memory_total::Csize_t
    type::ggml_backend_dev_type
    device_id::Ptr{Cchar}
    caps::ggml_backend_dev_caps
end

function ggml_backend_dev_name(device)
    ccall((:ggml_backend_dev_name, libllama), Ptr{Cchar}, (ggml_backend_dev_t,), device)
end

function ggml_backend_dev_description(device)
    ccall((:ggml_backend_dev_description, libllama), Ptr{Cchar}, (ggml_backend_dev_t,), device)
end

function ggml_backend_dev_memory(device, free, total)
    ccall((:ggml_backend_dev_memory, libllama), Cvoid, (ggml_backend_dev_t, Ptr{Csize_t}, Ptr{Csize_t}), device, free, total)
end

function ggml_backend_dev_type(device)
    ccall((:ggml_backend_dev_type, libllama), ggml_backend_dev_type, (ggml_backend_dev_t,), device)
end

function ggml_backend_dev_get_props(device, props)
    ccall((:ggml_backend_dev_get_props, libllama), Cvoid, (ggml_backend_dev_t, Ptr{ggml_backend_dev_props}), device, props)
end

function ggml_backend_dev_backend_reg(device)
    ccall((:ggml_backend_dev_backend_reg, libllama), ggml_backend_reg_t, (ggml_backend_dev_t,), device)
end

function ggml_backend_dev_init(device, params)
    ccall((:ggml_backend_dev_init, libllama), ggml_backend_t, (ggml_backend_dev_t, Ptr{Cchar}), device, params)
end

function ggml_backend_dev_buffer_type(device)
    ccall((:ggml_backend_dev_buffer_type, libllama), ggml_backend_buffer_type_t, (ggml_backend_dev_t,), device)
end

function ggml_backend_dev_host_buffer_type(device)
    ccall((:ggml_backend_dev_host_buffer_type, libllama), ggml_backend_buffer_type_t, (ggml_backend_dev_t,), device)
end

function ggml_backend_dev_buffer_from_host_ptr(device, ptr, size, max_tensor_size)
    ccall((:ggml_backend_dev_buffer_from_host_ptr, libllama), ggml_backend_buffer_t, (ggml_backend_dev_t, Ptr{Cvoid}, Csize_t, Csize_t), device, ptr, size, max_tensor_size)
end

function ggml_backend_dev_supports_op(device, op)
    ccall((:ggml_backend_dev_supports_op, libllama), Bool, (ggml_backend_dev_t, Ptr{ggml_tensor}), device, op)
end

function ggml_backend_dev_supports_buft(device, buft)
    ccall((:ggml_backend_dev_supports_buft, libllama), Bool, (ggml_backend_dev_t, ggml_backend_buffer_type_t), device, buft)
end

function ggml_backend_dev_offload_op(device, op)
    ccall((:ggml_backend_dev_offload_op, libllama), Bool, (ggml_backend_dev_t, Ptr{ggml_tensor}), device, op)
end

function ggml_backend_reg_name(reg)
    ccall((:ggml_backend_reg_name, libllama), Ptr{Cchar}, (ggml_backend_reg_t,), reg)
end

function ggml_backend_reg_dev_count(reg)
    ccall((:ggml_backend_reg_dev_count, libllama), Csize_t, (ggml_backend_reg_t,), reg)
end

function ggml_backend_reg_dev_get(reg, index)
    ccall((:ggml_backend_reg_dev_get, libllama), ggml_backend_dev_t, (ggml_backend_reg_t, Csize_t), reg, index)
end

function ggml_backend_reg_get_proc_address(reg, name)
    ccall((:ggml_backend_reg_get_proc_address, libllama), Ptr{Cvoid}, (ggml_backend_reg_t, Ptr{Cchar}), reg, name)
end

# typedef void * ( * ggml_backend_comm_init_t ) ( ggml_backend_t * backends , size_t n_backends )
const ggml_backend_comm_init_t = Ptr{Cvoid}

# typedef void ( * ggml_backend_comm_free_t ) ( void * comm_ctx )
const ggml_backend_comm_free_t = Ptr{Cvoid}

# typedef bool ( * ggml_backend_comm_allreduce_tensor_t ) ( void * comm_ctx , struct ggml_tensor * * tensors )
const ggml_backend_comm_allreduce_tensor_t = Ptr{Cvoid}

# typedef ggml_backend_buffer_type_t ( * ggml_backend_split_buffer_type_t ) ( int main_device , const float * tensor_split )
const ggml_backend_split_buffer_type_t = Ptr{Cvoid}

# typedef void ( * ggml_backend_set_n_threads_t ) ( ggml_backend_t backend , int n_threads )
const ggml_backend_set_n_threads_t = Ptr{Cvoid}

# typedef ggml_backend_buffer_type_t * ( * ggml_backend_dev_get_extra_bufts_t ) ( ggml_backend_dev_t device )
const ggml_backend_dev_get_extra_bufts_t = Ptr{Cvoid}

# typedef void ( * ggml_backend_set_abort_callback_t ) ( ggml_backend_t backend , ggml_abort_callback abort_callback , void * abort_callback_data )
const ggml_backend_set_abort_callback_t = Ptr{Cvoid}

struct ggml_backend_feature
    name::Ptr{Cchar}
    value::Ptr{Cchar}
end

# typedef struct ggml_backend_feature * ( * ggml_backend_get_features_t ) ( ggml_backend_reg_t reg )
const ggml_backend_get_features_t = Ptr{Cvoid}

function ggml_backend_register(reg)
    ccall((:ggml_backend_register, libllama), Cvoid, (ggml_backend_reg_t,), reg)
end

function ggml_backend_device_register(device)
    ccall((:ggml_backend_device_register, libllama), Cvoid, (ggml_backend_dev_t,), device)
end

function ggml_backend_reg_count()
    ccall((:ggml_backend_reg_count, libllama), Csize_t, ())
end

function ggml_backend_reg_get(index)
    ccall((:ggml_backend_reg_get, libllama), ggml_backend_reg_t, (Csize_t,), index)
end

function ggml_backend_reg_by_name(name)
    ccall((:ggml_backend_reg_by_name, libllama), ggml_backend_reg_t, (Ptr{Cchar},), name)
end

function ggml_backend_dev_count()
    ccall((:ggml_backend_dev_count, libllama), Csize_t, ())
end

function ggml_backend_dev_get(index)
    ccall((:ggml_backend_dev_get, libllama), ggml_backend_dev_t, (Csize_t,), index)
end

function ggml_backend_dev_by_name(name)
    ccall((:ggml_backend_dev_by_name, libllama), ggml_backend_dev_t, (Ptr{Cchar},), name)
end

function ggml_backend_dev_by_type(type)
    ccall((:ggml_backend_dev_by_type, libllama), ggml_backend_dev_t, (ggml_backend_dev_type,), type)
end

function ggml_backend_init_by_name(name, params)
    ccall((:ggml_backend_init_by_name, libllama), ggml_backend_t, (Ptr{Cchar}, Ptr{Cchar}), name, params)
end

function ggml_backend_init_by_type(type, params)
    ccall((:ggml_backend_init_by_type, libllama), ggml_backend_t, (ggml_backend_dev_type, Ptr{Cchar}), type, params)
end

function ggml_backend_init_best()
    ccall((:ggml_backend_init_best, libllama), ggml_backend_t, ())
end

function ggml_backend_load(path)
    ccall((:ggml_backend_load, libllama), ggml_backend_reg_t, (Ptr{Cchar},), path)
end

function ggml_backend_unload(reg)
    ccall((:ggml_backend_unload, libllama), Cvoid, (ggml_backend_reg_t,), reg)
end

function ggml_backend_load_all()
    ccall((:ggml_backend_load_all, libllama), Cvoid, ())
end

function ggml_backend_load_all_from_path(dir_path)
    ccall((:ggml_backend_load_all_from_path, libllama), Cvoid, (Ptr{Cchar},), dir_path)
end

mutable struct ggml_backend_sched end

const ggml_backend_sched_t = Ptr{ggml_backend_sched}

# typedef bool ( * ggml_backend_sched_eval_callback ) ( struct ggml_tensor * t , bool ask , void * user_data )
const ggml_backend_sched_eval_callback = Ptr{Cvoid}

function ggml_backend_sched_new(backends, bufts, n_backends, graph_size, parallel, op_offload)
    ccall((:ggml_backend_sched_new, libllama), ggml_backend_sched_t, (Ptr{ggml_backend_t}, Ptr{ggml_backend_buffer_type_t}, Cint, Csize_t, Bool, Bool), backends, bufts, n_backends, graph_size, parallel, op_offload)
end

function ggml_backend_sched_free(sched)
    ccall((:ggml_backend_sched_free, libllama), Cvoid, (ggml_backend_sched_t,), sched)
end

function ggml_backend_sched_reserve_size(sched, measure_graph, sizes)
    ccall((:ggml_backend_sched_reserve_size, libllama), Cvoid, (ggml_backend_sched_t, Ptr{ggml_cgraph}, Ptr{Csize_t}), sched, measure_graph, sizes)
end

function ggml_backend_sched_reserve(sched, measure_graph)
    ccall((:ggml_backend_sched_reserve, libllama), Bool, (ggml_backend_sched_t, Ptr{ggml_cgraph}), sched, measure_graph)
end

function ggml_backend_sched_get_n_backends(sched)
    ccall((:ggml_backend_sched_get_n_backends, libllama), Cint, (ggml_backend_sched_t,), sched)
end

function ggml_backend_sched_get_backend(sched, i)
    ccall((:ggml_backend_sched_get_backend, libllama), ggml_backend_t, (ggml_backend_sched_t, Cint), sched, i)
end

function ggml_backend_sched_get_n_splits(sched)
    ccall((:ggml_backend_sched_get_n_splits, libllama), Cint, (ggml_backend_sched_t,), sched)
end

function ggml_backend_sched_get_n_copies(sched)
    ccall((:ggml_backend_sched_get_n_copies, libllama), Cint, (ggml_backend_sched_t,), sched)
end

function ggml_backend_sched_get_buffer_type(sched, backend)
    ccall((:ggml_backend_sched_get_buffer_type, libllama), ggml_backend_buffer_type_t, (ggml_backend_sched_t, ggml_backend_t), sched, backend)
end

function ggml_backend_sched_get_buffer_size(sched, backend)
    ccall((:ggml_backend_sched_get_buffer_size, libllama), Csize_t, (ggml_backend_sched_t, ggml_backend_t), sched, backend)
end

function ggml_backend_sched_set_tensor_backend(sched, node, backend)
    ccall((:ggml_backend_sched_set_tensor_backend, libllama), Cvoid, (ggml_backend_sched_t, Ptr{ggml_tensor}, ggml_backend_t), sched, node, backend)
end

function ggml_backend_sched_get_tensor_backend(sched, node)
    ccall((:ggml_backend_sched_get_tensor_backend, libllama), ggml_backend_t, (ggml_backend_sched_t, Ptr{ggml_tensor}), sched, node)
end

function ggml_backend_sched_split_graph(sched, graph)
    ccall((:ggml_backend_sched_split_graph, libllama), Cvoid, (ggml_backend_sched_t, Ptr{ggml_cgraph}), sched, graph)
end

function ggml_backend_sched_alloc_graph(sched, graph)
    ccall((:ggml_backend_sched_alloc_graph, libllama), Bool, (ggml_backend_sched_t, Ptr{ggml_cgraph}), sched, graph)
end

function ggml_backend_sched_graph_compute(sched, graph)
    ccall((:ggml_backend_sched_graph_compute, libllama), ggml_status, (ggml_backend_sched_t, Ptr{ggml_cgraph}), sched, graph)
end

function ggml_backend_sched_graph_compute_async(sched, graph)
    ccall((:ggml_backend_sched_graph_compute_async, libllama), ggml_status, (ggml_backend_sched_t, Ptr{ggml_cgraph}), sched, graph)
end

function ggml_backend_sched_synchronize(sched)
    ccall((:ggml_backend_sched_synchronize, libllama), Cvoid, (ggml_backend_sched_t,), sched)
end

function ggml_backend_sched_reset(sched)
    ccall((:ggml_backend_sched_reset, libllama), Cvoid, (ggml_backend_sched_t,), sched)
end

function ggml_backend_sched_set_eval_callback(sched, callback, user_data)
    ccall((:ggml_backend_sched_set_eval_callback, libllama), Cvoid, (ggml_backend_sched_t, ggml_backend_sched_eval_callback, Ptr{Cvoid}), sched, callback, user_data)
end

@cenum ggml_backend_meta_split_axis::UInt32 begin
    GGML_BACKEND_SPLIT_AXIS_0 = 0
    GGML_BACKEND_SPLIT_AXIS_1 = 1
    GGML_BACKEND_SPLIT_AXIS_2 = 2
    GGML_BACKEND_SPLIT_AXIS_3 = 3
    GGML_BACKEND_SPLIT_AXIS_MIRRORED = 10
    GGML_BACKEND_SPLIT_AXIS_PARTIAL = 11
    GGML_BACKEND_SPLIT_AXIS_NONE = 98
    GGML_BACKEND_SPLIT_AXIS_UNKNOWN = 99
end

function ggml_backend_meta_split_axis_name(split_axis)
    ccall((:ggml_backend_meta_split_axis_name, libllama), Ptr{Cchar}, (ggml_backend_meta_split_axis,), split_axis)
end

struct ggml_backend_meta_split_state
    axis::ggml_backend_meta_split_axis
    ne::NTuple{256, Int64}
    nr::NTuple{16, UInt32}
    n_segments::UInt32
end

# typedef struct ggml_backend_meta_split_state ( * ggml_backend_meta_get_split_state_t ) ( const struct ggml_tensor * tensor , void * userdata )
const ggml_backend_meta_get_split_state_t = Ptr{Cvoid}

function ggml_backend_meta_device(devs, n_devs, get_split_state, get_split_state_ud)
    ccall((:ggml_backend_meta_device, libllama), ggml_backend_dev_t, (Ptr{ggml_backend_dev_t}, Csize_t, ggml_backend_meta_get_split_state_t, Ptr{Cvoid}), devs, n_devs, get_split_state, get_split_state_ud)
end

struct ggml_backend_graph_copy
    buffer::ggml_backend_buffer_t
    ctx_allocated::Ptr{ggml_context}
    ctx_unallocated::Ptr{ggml_context}
    graph::Ptr{ggml_cgraph}
end

function ggml_backend_graph_copy(backend, graph)
    ccall((:ggml_backend_graph_copy, libllama), ggml_backend_graph_copy, (ggml_backend_t, Ptr{ggml_cgraph}), backend, graph)
end

function ggml_backend_graph_copy_free(copy)
    ccall((:ggml_backend_graph_copy_free, libllama), Cvoid, (ggml_backend_graph_copy,), copy)
end

# typedef bool ( * ggml_backend_eval_callback ) ( int node_index , struct ggml_tensor * t1 , struct ggml_tensor * t2 , void * user_data )
const ggml_backend_eval_callback = Ptr{Cvoid}

function ggml_backend_compare_graph_backend(backend1, backend2, graph, callback, user_data, test_nodes, num_test_nodes)
    ccall((:ggml_backend_compare_graph_backend, libllama), Bool, (ggml_backend_t, ggml_backend_t, Ptr{ggml_cgraph}, ggml_backend_eval_callback, Ptr{Cvoid}, Ptr{Ptr{ggml_tensor}}, Csize_t), backend1, backend2, graph, callback, user_data, test_nodes, num_test_nodes)
end

function ggml_backend_tensor_alloc(buffer, tensor, addr)
    ccall((:ggml_backend_tensor_alloc, libllama), ggml_status, (ggml_backend_buffer_t, Ptr{ggml_tensor}, Ptr{Cvoid}), buffer, tensor, addr)
end

function ggml_backend_view_init(tensor)
    ccall((:ggml_backend_view_init, libllama), ggml_status, (Ptr{ggml_tensor},), tensor)
end

function ggml_backend_cpu_buffer_from_ptr(ptr, size)
    ccall((:ggml_backend_cpu_buffer_from_ptr, libllama), ggml_backend_buffer_t, (Ptr{Cvoid}, Csize_t), ptr, size)
end

function ggml_backend_cpu_buffer_type()
    ccall((:ggml_backend_cpu_buffer_type, libllama), ggml_backend_buffer_type_t, ())
end

struct ggml_cplan
    work_size::Csize_t
    work_data::Ptr{UInt8}
    n_threads::Cint
    threadpool::Ptr{ggml_threadpool}
    abort_callback::ggml_abort_callback
    abort_callback_data::Ptr{Cvoid}
    use_ref::Bool
end

@cenum ggml_numa_strategy::UInt32 begin
    GGML_NUMA_STRATEGY_DISABLED = 0
    GGML_NUMA_STRATEGY_DISTRIBUTE = 1
    GGML_NUMA_STRATEGY_ISOLATE = 2
    GGML_NUMA_STRATEGY_NUMACTL = 3
    GGML_NUMA_STRATEGY_MIRROR = 4
    GGML_NUMA_STRATEGY_COUNT = 5
end

function ggml_numa_init(numa)
    ccall((:ggml_numa_init, libllama), Cvoid, (ggml_numa_strategy,), numa)
end

function ggml_is_numa()
    ccall((:ggml_is_numa, libllama), Bool, ())
end

function ggml_new_i32(ctx, value)
    ccall((:ggml_new_i32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Int32), ctx, value)
end

function ggml_new_f32(ctx, value)
    ccall((:ggml_new_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_context}, Cfloat), ctx, value)
end

function ggml_set_i32(tensor, value)
    ccall((:ggml_set_i32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_tensor}, Int32), tensor, value)
end

function ggml_set_f32(tensor, value)
    ccall((:ggml_set_f32, libllama), Ptr{ggml_tensor}, (Ptr{ggml_tensor}, Cfloat), tensor, value)
end

function ggml_get_i32_1d(tensor, i)
    ccall((:ggml_get_i32_1d, libllama), Int32, (Ptr{ggml_tensor}, Cint), tensor, i)
end

function ggml_set_i32_1d(tensor, i, value)
    ccall((:ggml_set_i32_1d, libllama), Cvoid, (Ptr{ggml_tensor}, Cint, Int32), tensor, i, value)
end

function ggml_get_i32_nd(tensor, i0, i1, i2, i3)
    ccall((:ggml_get_i32_nd, libllama), Int32, (Ptr{ggml_tensor}, Cint, Cint, Cint, Cint), tensor, i0, i1, i2, i3)
end

function ggml_set_i32_nd(tensor, i0, i1, i2, i3, value)
    ccall((:ggml_set_i32_nd, libllama), Cvoid, (Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Int32), tensor, i0, i1, i2, i3, value)
end

function ggml_get_f32_1d(tensor, i)
    ccall((:ggml_get_f32_1d, libllama), Cfloat, (Ptr{ggml_tensor}, Cint), tensor, i)
end

function ggml_set_f32_1d(tensor, i, value)
    ccall((:ggml_set_f32_1d, libllama), Cvoid, (Ptr{ggml_tensor}, Cint, Cfloat), tensor, i, value)
end

function ggml_get_f32_nd(tensor, i0, i1, i2, i3)
    ccall((:ggml_get_f32_nd, libllama), Cfloat, (Ptr{ggml_tensor}, Cint, Cint, Cint, Cint), tensor, i0, i1, i2, i3)
end

function ggml_set_f32_nd(tensor, i0, i1, i2, i3, value)
    ccall((:ggml_set_f32_nd, libllama), Cvoid, (Ptr{ggml_tensor}, Cint, Cint, Cint, Cint, Cfloat), tensor, i0, i1, i2, i3, value)
end

function ggml_threadpool_new(params)
    ccall((:ggml_threadpool_new, libllama), Ptr{ggml_threadpool}, (Ptr{ggml_threadpool_params},), params)
end

function ggml_threadpool_free(threadpool)
    ccall((:ggml_threadpool_free, libllama), Cvoid, (Ptr{ggml_threadpool},), threadpool)
end

function ggml_threadpool_get_n_threads(threadpool)
    ccall((:ggml_threadpool_get_n_threads, libllama), Cint, (Ptr{ggml_threadpool},), threadpool)
end

function ggml_threadpool_pause(threadpool)
    ccall((:ggml_threadpool_pause, libllama), Cvoid, (Ptr{ggml_threadpool},), threadpool)
end

function ggml_threadpool_resume(threadpool)
    ccall((:ggml_threadpool_resume, libllama), Cvoid, (Ptr{ggml_threadpool},), threadpool)
end

function ggml_graph_plan(cgraph, n_threads, threadpool)
    ccall((:ggml_graph_plan, libllama), ggml_cplan, (Ptr{ggml_cgraph}, Cint, Ptr{ggml_threadpool}), cgraph, n_threads, threadpool)
end

function ggml_graph_compute(cgraph, cplan)
    ccall((:ggml_graph_compute, libllama), ggml_status, (Ptr{ggml_cgraph}, Ptr{ggml_cplan}), cgraph, cplan)
end

function ggml_graph_compute_with_ctx(ctx, cgraph, n_threads)
    ccall((:ggml_graph_compute_with_ctx, libllama), ggml_status, (Ptr{ggml_context}, Ptr{ggml_cgraph}, Cint), ctx, cgraph, n_threads)
end

function ggml_cpu_has_sse3()
    ccall((:ggml_cpu_has_sse3, libllama), Cint, ())
end

function ggml_cpu_has_ssse3()
    ccall((:ggml_cpu_has_ssse3, libllama), Cint, ())
end

function ggml_cpu_has_avx()
    ccall((:ggml_cpu_has_avx, libllama), Cint, ())
end

function ggml_cpu_has_avx_vnni()
    ccall((:ggml_cpu_has_avx_vnni, libllama), Cint, ())
end

function ggml_cpu_has_avx2()
    ccall((:ggml_cpu_has_avx2, libllama), Cint, ())
end

function ggml_cpu_has_bmi2()
    ccall((:ggml_cpu_has_bmi2, libllama), Cint, ())
end

function ggml_cpu_has_f16c()
    ccall((:ggml_cpu_has_f16c, libllama), Cint, ())
end

function ggml_cpu_has_fma()
    ccall((:ggml_cpu_has_fma, libllama), Cint, ())
end

function ggml_cpu_has_avx512()
    ccall((:ggml_cpu_has_avx512, libllama), Cint, ())
end

function ggml_cpu_has_avx512_vbmi()
    ccall((:ggml_cpu_has_avx512_vbmi, libllama), Cint, ())
end

function ggml_cpu_has_avx512_vnni()
    ccall((:ggml_cpu_has_avx512_vnni, libllama), Cint, ())
end

function ggml_cpu_has_avx512_bf16()
    ccall((:ggml_cpu_has_avx512_bf16, libllama), Cint, ())
end

function ggml_cpu_has_amx_int8()
    ccall((:ggml_cpu_has_amx_int8, libllama), Cint, ())
end

function ggml_cpu_has_neon()
    ccall((:ggml_cpu_has_neon, libllama), Cint, ())
end

function ggml_cpu_has_arm_fma()
    ccall((:ggml_cpu_has_arm_fma, libllama), Cint, ())
end

function ggml_cpu_has_fp16_va()
    ccall((:ggml_cpu_has_fp16_va, libllama), Cint, ())
end

function ggml_cpu_has_dotprod()
    ccall((:ggml_cpu_has_dotprod, libllama), Cint, ())
end

function ggml_cpu_has_matmul_int8()
    ccall((:ggml_cpu_has_matmul_int8, libllama), Cint, ())
end

function ggml_cpu_has_sve()
    ccall((:ggml_cpu_has_sve, libllama), Cint, ())
end

function ggml_cpu_get_sve_cnt()
    ccall((:ggml_cpu_get_sve_cnt, libllama), Cint, ())
end

function ggml_cpu_has_sme()
    ccall((:ggml_cpu_has_sme, libllama), Cint, ())
end

function ggml_cpu_has_sme2()
    ccall((:ggml_cpu_has_sme2, libllama), Cint, ())
end

function ggml_cpu_has_riscv_v()
    ccall((:ggml_cpu_has_riscv_v, libllama), Cint, ())
end

function ggml_cpu_get_rvv_vlen()
    ccall((:ggml_cpu_get_rvv_vlen, libllama), Cint, ())
end

function ggml_cpu_has_vsx()
    ccall((:ggml_cpu_has_vsx, libllama), Cint, ())
end

function ggml_cpu_has_vxe()
    ccall((:ggml_cpu_has_vxe, libllama), Cint, ())
end

function ggml_cpu_has_wasm_simd()
    ccall((:ggml_cpu_has_wasm_simd, libllama), Cint, ())
end

function ggml_cpu_has_llamafile()
    ccall((:ggml_cpu_has_llamafile, libllama), Cint, ())
end

# typedef void ( * ggml_vec_dot_t ) ( int n , float * GGML_RESTRICT s , size_t bs , const void * GGML_RESTRICT x , size_t bx , const void * GGML_RESTRICT y , size_t by , int nrc )
const ggml_vec_dot_t = Ptr{Cvoid}

struct ggml_type_traits_cpu
    from_float::ggml_from_float_t
    vec_dot::ggml_vec_dot_t
    vec_dot_type::ggml_type
    nrows::Int64
end

function ggml_get_type_traits_cpu(type)
    ccall((:ggml_get_type_traits_cpu, libllama), Ptr{ggml_type_traits_cpu}, (ggml_type,), type)
end

function ggml_cpu_init()
    ccall((:ggml_cpu_init, libllama), Cvoid, ())
end

function ggml_backend_cpu_init()
    ccall((:ggml_backend_cpu_init, libllama), ggml_backend_t, ())
end

function ggml_backend_is_cpu(backend)
    ccall((:ggml_backend_is_cpu, libllama), Bool, (ggml_backend_t,), backend)
end

function ggml_backend_cpu_set_n_threads(backend_cpu, n_threads)
    ccall((:ggml_backend_cpu_set_n_threads, libllama), Cvoid, (ggml_backend_t, Cint), backend_cpu, n_threads)
end

function ggml_backend_cpu_set_threadpool(backend_cpu, threadpool)
    ccall((:ggml_backend_cpu_set_threadpool, libllama), Cvoid, (ggml_backend_t, ggml_threadpool_t), backend_cpu, threadpool)
end

function ggml_backend_cpu_set_abort_callback(backend_cpu, abort_callback, abort_callback_data)
    ccall((:ggml_backend_cpu_set_abort_callback, libllama), Cvoid, (ggml_backend_t, ggml_abort_callback, Ptr{Cvoid}), backend_cpu, abort_callback, abort_callback_data)
end

function ggml_backend_cpu_set_use_ref(backend_cpu, use_ref)
    ccall((:ggml_backend_cpu_set_use_ref, libllama), Cvoid, (ggml_backend_t, Bool), backend_cpu, use_ref)
end

function ggml_backend_cpu_reg()
    ccall((:ggml_backend_cpu_reg, libllama), ggml_backend_reg_t, ())
end

function ggml_cpu_fp32_to_fp32(arg1, arg2, arg3)
    ccall((:ggml_cpu_fp32_to_fp32, libllama), Cvoid, (Ptr{Cfloat}, Ptr{Cfloat}, Int64), arg1, arg2, arg3)
end

function ggml_cpu_fp32_to_i32(arg1, arg2, arg3)
    ccall((:ggml_cpu_fp32_to_i32, libllama), Cvoid, (Ptr{Cfloat}, Ptr{Int32}, Int64), arg1, arg2, arg3)
end

function ggml_cpu_fp32_to_fp16(arg1, arg2, arg3)
    ccall((:ggml_cpu_fp32_to_fp16, libllama), Cvoid, (Ptr{Cfloat}, Ptr{ggml_fp16_t}, Int64), arg1, arg2, arg3)
end

function ggml_cpu_fp16_to_fp32(arg1, arg2, arg3)
    ccall((:ggml_cpu_fp16_to_fp32, libllama), Cvoid, (Ptr{ggml_fp16_t}, Ptr{Cfloat}, Int64), arg1, arg2, arg3)
end

function ggml_cpu_fp32_to_bf16(arg1, arg2, arg3)
    ccall((:ggml_cpu_fp32_to_bf16, libllama), Cvoid, (Ptr{Cfloat}, Ptr{ggml_bf16_t}, Int64), arg1, arg2, arg3)
end

function ggml_cpu_bf16_to_fp32(arg1, arg2, arg3)
    ccall((:ggml_cpu_bf16_to_fp32, libllama), Cvoid, (Ptr{ggml_bf16_t}, Ptr{Cfloat}, Int64), arg1, arg2, arg3)
end

mutable struct ggml_opt_dataset end

mutable struct ggml_opt_context end

mutable struct ggml_opt_result end

const ggml_opt_dataset_t = Ptr{ggml_opt_dataset}

const ggml_opt_context_t = Ptr{ggml_opt_context}

const ggml_opt_result_t = Ptr{ggml_opt_result}

@cenum ggml_opt_loss_type::UInt32 begin
    GGML_OPT_LOSS_TYPE_MEAN = 0
    GGML_OPT_LOSS_TYPE_SUM = 1
    GGML_OPT_LOSS_TYPE_CROSS_ENTROPY = 2
    GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR = 3
end

function ggml_opt_dataset_init(type_data, type_label, ne_datapoint, ne_label, ndata, ndata_shard)
    ccall((:ggml_opt_dataset_init, libllama), ggml_opt_dataset_t, (ggml_type, ggml_type, Int64, Int64, Int64, Int64), type_data, type_label, ne_datapoint, ne_label, ndata, ndata_shard)
end

function ggml_opt_dataset_free(dataset)
    ccall((:ggml_opt_dataset_free, libllama), Cvoid, (ggml_opt_dataset_t,), dataset)
end

function ggml_opt_dataset_ndata(dataset)
    ccall((:ggml_opt_dataset_ndata, libllama), Int64, (ggml_opt_dataset_t,), dataset)
end

function ggml_opt_dataset_data(dataset)
    ccall((:ggml_opt_dataset_data, libllama), Ptr{ggml_tensor}, (ggml_opt_dataset_t,), dataset)
end

function ggml_opt_dataset_labels(dataset)
    ccall((:ggml_opt_dataset_labels, libllama), Ptr{ggml_tensor}, (ggml_opt_dataset_t,), dataset)
end

function ggml_opt_dataset_shuffle(opt_ctx, dataset, idata)
    ccall((:ggml_opt_dataset_shuffle, libllama), Cvoid, (ggml_opt_context_t, ggml_opt_dataset_t, Int64), opt_ctx, dataset, idata)
end

function ggml_opt_dataset_get_batch(dataset, data_batch, labels_batch, ibatch)
    ccall((:ggml_opt_dataset_get_batch, libllama), Cvoid, (ggml_opt_dataset_t, Ptr{ggml_tensor}, Ptr{ggml_tensor}, Int64), dataset, data_batch, labels_batch, ibatch)
end

function ggml_opt_dataset_get_batch_host(dataset, data_batch, nb_data_batch, labels_batch, ibatch)
    ccall((:ggml_opt_dataset_get_batch_host, libllama), Cvoid, (ggml_opt_dataset_t, Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Int64), dataset, data_batch, nb_data_batch, labels_batch, ibatch)
end

@cenum ggml_opt_build_type::UInt32 begin
    GGML_OPT_BUILD_TYPE_FORWARD = 10
    GGML_OPT_BUILD_TYPE_GRAD = 20
    GGML_OPT_BUILD_TYPE_OPT = 30
end

@cenum ggml_opt_optimizer_type::UInt32 begin
    GGML_OPT_OPTIMIZER_TYPE_ADAMW = 0
    GGML_OPT_OPTIMIZER_TYPE_SGD = 1
    GGML_OPT_OPTIMIZER_TYPE_COUNT = 2
end

struct var"##Ctag#277"
    alpha::Cfloat
    beta1::Cfloat
    beta2::Cfloat
    eps::Cfloat
    wd::Cfloat
end
function Base.getproperty(x::Ptr{var"##Ctag#277"}, f::Symbol)
    f === :alpha && return Ptr{Cfloat}(x + 0)
    f === :beta1 && return Ptr{Cfloat}(x + 4)
    f === :beta2 && return Ptr{Cfloat}(x + 8)
    f === :eps && return Ptr{Cfloat}(x + 12)
    f === :wd && return Ptr{Cfloat}(x + 16)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#277", f::Symbol)
    r = Ref{var"##Ctag#277"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#277"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#277"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct var"##Ctag#278"
    alpha::Cfloat
    wd::Cfloat
end
function Base.getproperty(x::Ptr{var"##Ctag#278"}, f::Symbol)
    f === :alpha && return Ptr{Cfloat}(x + 0)
    f === :wd && return Ptr{Cfloat}(x + 4)
    return getfield(x, f)
end

function Base.getproperty(x::var"##Ctag#278", f::Symbol)
    r = Ref{var"##Ctag#278"}(x)
    ptr = Base.unsafe_convert(Ptr{var"##Ctag#278"}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{var"##Ctag#278"}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end


struct ggml_opt_optimizer_params
    data::NTuple{28, UInt8}
end

function Base.getproperty(x::Ptr{ggml_opt_optimizer_params}, f::Symbol)
    f === :adamw && return Ptr{var"##Ctag#277"}(x + 0)
    f === :sgd && return Ptr{var"##Ctag#278"}(x + 20)
    return getfield(x, f)
end

function Base.getproperty(x::ggml_opt_optimizer_params, f::Symbol)
    r = Ref{ggml_opt_optimizer_params}(x)
    ptr = Base.unsafe_convert(Ptr{ggml_opt_optimizer_params}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{ggml_opt_optimizer_params}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::ggml_opt_optimizer_params, private::Bool = false)
    (:adamw, :sgd, if private
            fieldnames(typeof(x))
        else
            ()
        end...)
end

# typedef struct ggml_opt_optimizer_params ( * ggml_opt_get_optimizer_params ) ( void * userdata )
const ggml_opt_get_optimizer_params = Ptr{Cvoid}

function ggml_opt_get_default_optimizer_params(userdata)
    ccall((:ggml_opt_get_default_optimizer_params, libllama), ggml_opt_optimizer_params, (Ptr{Cvoid},), userdata)
end

function ggml_opt_get_constant_optimizer_params(userdata)
    ccall((:ggml_opt_get_constant_optimizer_params, libllama), ggml_opt_optimizer_params, (Ptr{Cvoid},), userdata)
end

struct ggml_opt_params
    backend_sched::ggml_backend_sched_t
    ctx_compute::Ptr{ggml_context}
    inputs::Ptr{ggml_tensor}
    outputs::Ptr{ggml_tensor}
    loss_type::ggml_opt_loss_type
    build_type::ggml_opt_build_type
    opt_period::Int32
    get_opt_pars::ggml_opt_get_optimizer_params
    get_opt_pars_ud::Ptr{Cvoid}
    optimizer::ggml_opt_optimizer_type
end

function ggml_opt_default_params(backend_sched, loss_type)
    ccall((:ggml_opt_default_params, libllama), ggml_opt_params, (ggml_backend_sched_t, ggml_opt_loss_type), backend_sched, loss_type)
end

function ggml_opt_init(params)
    ccall((:ggml_opt_init, libllama), ggml_opt_context_t, (ggml_opt_params,), params)
end

function ggml_opt_free(opt_ctx)
    ccall((:ggml_opt_free, libllama), Cvoid, (ggml_opt_context_t,), opt_ctx)
end

function ggml_opt_reset(opt_ctx, optimizer)
    ccall((:ggml_opt_reset, libllama), Cvoid, (ggml_opt_context_t, Bool), opt_ctx, optimizer)
end

function ggml_opt_static_graphs(opt_ctx)
    ccall((:ggml_opt_static_graphs, libllama), Bool, (ggml_opt_context_t,), opt_ctx)
end

function ggml_opt_inputs(opt_ctx)
    ccall((:ggml_opt_inputs, libllama), Ptr{ggml_tensor}, (ggml_opt_context_t,), opt_ctx)
end

function ggml_opt_outputs(opt_ctx)
    ccall((:ggml_opt_outputs, libllama), Ptr{ggml_tensor}, (ggml_opt_context_t,), opt_ctx)
end

function ggml_opt_labels(opt_ctx)
    ccall((:ggml_opt_labels, libllama), Ptr{ggml_tensor}, (ggml_opt_context_t,), opt_ctx)
end

function ggml_opt_loss(opt_ctx)
    ccall((:ggml_opt_loss, libllama), Ptr{ggml_tensor}, (ggml_opt_context_t,), opt_ctx)
end

function ggml_opt_pred(opt_ctx)
    ccall((:ggml_opt_pred, libllama), Ptr{ggml_tensor}, (ggml_opt_context_t,), opt_ctx)
end

function ggml_opt_ncorrect(opt_ctx)
    ccall((:ggml_opt_ncorrect, libllama), Ptr{ggml_tensor}, (ggml_opt_context_t,), opt_ctx)
end

function ggml_opt_grad_acc(opt_ctx, node)
    ccall((:ggml_opt_grad_acc, libllama), Ptr{ggml_tensor}, (ggml_opt_context_t, Ptr{ggml_tensor}), opt_ctx, node)
end

function ggml_opt_context_optimizer_type(arg1)
    ccall((:ggml_opt_context_optimizer_type, libllama), ggml_opt_optimizer_type, (ggml_opt_context_t,), arg1)
end

function ggml_opt_optimizer_name(arg1)
    ccall((:ggml_opt_optimizer_name, libllama), Ptr{Cchar}, (ggml_opt_optimizer_type,), arg1)
end

function ggml_opt_result_init()
    ccall((:ggml_opt_result_init, libllama), ggml_opt_result_t, ())
end

function ggml_opt_result_free(result)
    ccall((:ggml_opt_result_free, libllama), Cvoid, (ggml_opt_result_t,), result)
end

function ggml_opt_result_reset(result)
    ccall((:ggml_opt_result_reset, libllama), Cvoid, (ggml_opt_result_t,), result)
end

function ggml_opt_result_ndata(result, ndata)
    ccall((:ggml_opt_result_ndata, libllama), Cvoid, (ggml_opt_result_t, Ptr{Int64}), result, ndata)
end

function ggml_opt_result_loss(result, loss, unc)
    ccall((:ggml_opt_result_loss, libllama), Cvoid, (ggml_opt_result_t, Ptr{Cdouble}, Ptr{Cdouble}), result, loss, unc)
end

function ggml_opt_result_pred(result, pred)
    ccall((:ggml_opt_result_pred, libllama), Cvoid, (ggml_opt_result_t, Ptr{Int32}), result, pred)
end

function ggml_opt_result_accuracy(result, accuracy, unc)
    ccall((:ggml_opt_result_accuracy, libllama), Cvoid, (ggml_opt_result_t, Ptr{Cdouble}, Ptr{Cdouble}), result, accuracy, unc)
end

function ggml_opt_prepare_alloc(opt_ctx, ctx_compute, gf, inputs, outputs)
    ccall((:ggml_opt_prepare_alloc, libllama), Cvoid, (ggml_opt_context_t, Ptr{ggml_context}, Ptr{ggml_cgraph}, Ptr{ggml_tensor}, Ptr{ggml_tensor}), opt_ctx, ctx_compute, gf, inputs, outputs)
end

function ggml_opt_alloc(opt_ctx, backward)
    ccall((:ggml_opt_alloc, libllama), Cvoid, (ggml_opt_context_t, Bool), opt_ctx, backward)
end

function ggml_opt_eval(opt_ctx, result)
    ccall((:ggml_opt_eval, libllama), Cvoid, (ggml_opt_context_t, ggml_opt_result_t), opt_ctx, result)
end

# typedef void ( * ggml_opt_epoch_callback ) ( bool train , // true after training evaluation, false after validation evaluation ggml_opt_context_t opt_ctx , ggml_opt_dataset_t dataset , ggml_opt_result_t result , // result associated with the dataset subsection int64_t ibatch , // number of batches that have been evaluated so far int64_t ibatch_max , // total number of batches in this dataset subsection int64_t t_start_us )
const ggml_opt_epoch_callback = Ptr{Cvoid}

function ggml_opt_epoch(opt_ctx, dataset, result_train, result_eval, idata_split, callback_train, callback_eval)
    ccall((:ggml_opt_epoch, libllama), Cvoid, (ggml_opt_context_t, ggml_opt_dataset_t, ggml_opt_result_t, ggml_opt_result_t, Int64, ggml_opt_epoch_callback, ggml_opt_epoch_callback), opt_ctx, dataset, result_train, result_eval, idata_split, callback_train, callback_eval)
end

function ggml_opt_epoch_callback_progress_bar(train, opt_ctx, dataset, result, ibatch, ibatch_max, t_start_us)
    ccall((:ggml_opt_epoch_callback_progress_bar, libllama), Cvoid, (Bool, ggml_opt_context_t, ggml_opt_dataset_t, ggml_opt_result_t, Int64, Int64, Int64), train, opt_ctx, dataset, result, ibatch, ibatch_max, t_start_us)
end

function ggml_opt_fit(backend_sched, ctx_compute, inputs, outputs, dataset, loss_type, optimizer, get_opt_pars, nepoch, nbatch_logical, val_split, silent)
    ccall((:ggml_opt_fit, libllama), Cvoid, (ggml_backend_sched_t, Ptr{ggml_context}, Ptr{ggml_tensor}, Ptr{ggml_tensor}, ggml_opt_dataset_t, ggml_opt_loss_type, ggml_opt_optimizer_type, ggml_opt_get_optimizer_params, Int64, Int64, Cfloat, Bool), backend_sched, ctx_compute, inputs, outputs, dataset, loss_type, optimizer, get_opt_pars, nepoch, nbatch_logical, val_split, silent)
end

@cenum gguf_type::UInt32 begin
    GGUF_TYPE_UINT8 = 0
    GGUF_TYPE_INT8 = 1
    GGUF_TYPE_UINT16 = 2
    GGUF_TYPE_INT16 = 3
    GGUF_TYPE_UINT32 = 4
    GGUF_TYPE_INT32 = 5
    GGUF_TYPE_FLOAT32 = 6
    GGUF_TYPE_BOOL = 7
    GGUF_TYPE_STRING = 8
    GGUF_TYPE_ARRAY = 9
    GGUF_TYPE_UINT64 = 10
    GGUF_TYPE_INT64 = 11
    GGUF_TYPE_FLOAT64 = 12
    GGUF_TYPE_COUNT = 13
end

mutable struct gguf_context end

struct gguf_init_params
    no_alloc::Bool
    ctx::Ptr{Ptr{ggml_context}}
end

# typedef size_t ( * gguf_reader_callback_t ) ( void * userdata , void * output , uint64_t offset , size_t len )
const gguf_reader_callback_t = Ptr{Cvoid}

function gguf_init_empty()
    ccall((:gguf_init_empty, libllama), Ptr{gguf_context}, ())
end

function gguf_init_from_file_ptr(file, params)
    ccall((:gguf_init_from_file_ptr, libllama), Ptr{gguf_context}, (Ptr{Libc.FILE}, gguf_init_params), file, params)
end

function gguf_init_from_file(fname, params)
    ccall((:gguf_init_from_file, libllama), Ptr{gguf_context}, (Ptr{Cchar}, gguf_init_params), fname, params)
end

function gguf_init_from_buffer(data, size, params)
    ccall((:gguf_init_from_buffer, libllama), Ptr{gguf_context}, (Ptr{Cvoid}, Csize_t, gguf_init_params), data, size, params)
end

function gguf_init_from_callback(callback, userdata, max_chunk_read, max_expected_size, params)
    ccall((:gguf_init_from_callback, libllama), Ptr{gguf_context}, (gguf_reader_callback_t, Ptr{Cvoid}, Csize_t, UInt64, gguf_init_params), callback, userdata, max_chunk_read, max_expected_size, params)
end

function gguf_free(ctx)
    ccall((:gguf_free, libllama), Cvoid, (Ptr{gguf_context},), ctx)
end

function gguf_type_name(type)
    ccall((:gguf_type_name, libllama), Ptr{Cchar}, (gguf_type,), type)
end

function gguf_get_version(ctx)
    ccall((:gguf_get_version, libllama), UInt32, (Ptr{gguf_context},), ctx)
end

function gguf_get_alignment(ctx)
    ccall((:gguf_get_alignment, libllama), Csize_t, (Ptr{gguf_context},), ctx)
end

function gguf_get_data_offset(ctx)
    ccall((:gguf_get_data_offset, libllama), Csize_t, (Ptr{gguf_context},), ctx)
end

function gguf_get_n_kv(ctx)
    ccall((:gguf_get_n_kv, libllama), Int64, (Ptr{gguf_context},), ctx)
end

function gguf_find_key(ctx, key)
    ccall((:gguf_find_key, libllama), Int64, (Ptr{gguf_context}, Ptr{Cchar}), ctx, key)
end

function gguf_get_key(ctx, key_id)
    ccall((:gguf_get_key, libllama), Ptr{Cchar}, (Ptr{gguf_context}, Int64), ctx, key_id)
end

function gguf_get_kv_type(ctx, key_id)
    ccall((:gguf_get_kv_type, libllama), gguf_type, (Ptr{gguf_context}, Int64), ctx, key_id)
end

function gguf_get_arr_type(ctx, key_id)
    ccall((:gguf_get_arr_type, libllama), gguf_type, (Ptr{gguf_context}, Int64), ctx, key_id)
end

function gguf_get_val_u8(ctx, key_id)
    ccall((:gguf_get_val_u8, libllama), UInt8, (Ptr{gguf_context}, Int64), ctx, key_id)
end

function gguf_get_val_i8(ctx, key_id)
    ccall((:gguf_get_val_i8, libllama), Int8, (Ptr{gguf_context}, Int64), ctx, key_id)
end

function gguf_get_val_u16(ctx, key_id)
    ccall((:gguf_get_val_u16, libllama), UInt16, (Ptr{gguf_context}, Int64), ctx, key_id)
end

function gguf_get_val_i16(ctx, key_id)
    ccall((:gguf_get_val_i16, libllama), Int16, (Ptr{gguf_context}, Int64), ctx, key_id)
end

function gguf_get_val_u32(ctx, key_id)
    ccall((:gguf_get_val_u32, libllama), UInt32, (Ptr{gguf_context}, Int64), ctx, key_id)
end

function gguf_get_val_i32(ctx, key_id)
    ccall((:gguf_get_val_i32, libllama), Int32, (Ptr{gguf_context}, Int64), ctx, key_id)
end

function gguf_get_val_f32(ctx, key_id)
    ccall((:gguf_get_val_f32, libllama), Cfloat, (Ptr{gguf_context}, Int64), ctx, key_id)
end

function gguf_get_val_u64(ctx, key_id)
    ccall((:gguf_get_val_u64, libllama), UInt64, (Ptr{gguf_context}, Int64), ctx, key_id)
end

function gguf_get_val_i64(ctx, key_id)
    ccall((:gguf_get_val_i64, libllama), Int64, (Ptr{gguf_context}, Int64), ctx, key_id)
end

function gguf_get_val_f64(ctx, key_id)
    ccall((:gguf_get_val_f64, libllama), Cdouble, (Ptr{gguf_context}, Int64), ctx, key_id)
end

function gguf_get_val_bool(ctx, key_id)
    ccall((:gguf_get_val_bool, libllama), Bool, (Ptr{gguf_context}, Int64), ctx, key_id)
end

function gguf_get_val_str(ctx, key_id)
    ccall((:gguf_get_val_str, libllama), Ptr{Cchar}, (Ptr{gguf_context}, Int64), ctx, key_id)
end

function gguf_get_val_data(ctx, key_id)
    ccall((:gguf_get_val_data, libllama), Ptr{Cvoid}, (Ptr{gguf_context}, Int64), ctx, key_id)
end

function gguf_get_arr_n(ctx, key_id)
    ccall((:gguf_get_arr_n, libllama), Csize_t, (Ptr{gguf_context}, Int64), ctx, key_id)
end

function gguf_get_arr_data(ctx, key_id)
    ccall((:gguf_get_arr_data, libllama), Ptr{Cvoid}, (Ptr{gguf_context}, Int64), ctx, key_id)
end

function gguf_get_arr_str(ctx, key_id, i)
    ccall((:gguf_get_arr_str, libllama), Ptr{Cchar}, (Ptr{gguf_context}, Int64, Csize_t), ctx, key_id, i)
end

function gguf_get_n_tensors(ctx)
    ccall((:gguf_get_n_tensors, libllama), Int64, (Ptr{gguf_context},), ctx)
end

function gguf_find_tensor(ctx, name)
    ccall((:gguf_find_tensor, libllama), Int64, (Ptr{gguf_context}, Ptr{Cchar}), ctx, name)
end

function gguf_get_tensor_offset(ctx, tensor_id)
    ccall((:gguf_get_tensor_offset, libllama), Csize_t, (Ptr{gguf_context}, Int64), ctx, tensor_id)
end

function gguf_get_tensor_name(ctx, tensor_id)
    ccall((:gguf_get_tensor_name, libllama), Ptr{Cchar}, (Ptr{gguf_context}, Int64), ctx, tensor_id)
end

function gguf_get_tensor_ne(ctx, tensor_id)
    ccall((:gguf_get_tensor_ne, libllama), Ptr{Int64}, (Ptr{gguf_context}, Int64), ctx, tensor_id)
end

function gguf_get_tensor_type(ctx, tensor_id)
    ccall((:gguf_get_tensor_type, libllama), ggml_type, (Ptr{gguf_context}, Int64), ctx, tensor_id)
end

function gguf_get_tensor_size(ctx, tensor_id)
    ccall((:gguf_get_tensor_size, libllama), Csize_t, (Ptr{gguf_context}, Int64), ctx, tensor_id)
end

function gguf_remove_key(ctx, key)
    ccall((:gguf_remove_key, libllama), Int64, (Ptr{gguf_context}, Ptr{Cchar}), ctx, key)
end

function gguf_set_val_u8(ctx, key, val)
    ccall((:gguf_set_val_u8, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, UInt8), ctx, key, val)
end

function gguf_set_val_i8(ctx, key, val)
    ccall((:gguf_set_val_i8, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, Int8), ctx, key, val)
end

function gguf_set_val_u16(ctx, key, val)
    ccall((:gguf_set_val_u16, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, UInt16), ctx, key, val)
end

function gguf_set_val_i16(ctx, key, val)
    ccall((:gguf_set_val_i16, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, Int16), ctx, key, val)
end

function gguf_set_val_u32(ctx, key, val)
    ccall((:gguf_set_val_u32, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, UInt32), ctx, key, val)
end

function gguf_set_val_i32(ctx, key, val)
    ccall((:gguf_set_val_i32, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, Int32), ctx, key, val)
end

function gguf_set_val_f32(ctx, key, val)
    ccall((:gguf_set_val_f32, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, Cfloat), ctx, key, val)
end

function gguf_set_val_u64(ctx, key, val)
    ccall((:gguf_set_val_u64, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, UInt64), ctx, key, val)
end

function gguf_set_val_i64(ctx, key, val)
    ccall((:gguf_set_val_i64, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, Int64), ctx, key, val)
end

function gguf_set_val_f64(ctx, key, val)
    ccall((:gguf_set_val_f64, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, Cdouble), ctx, key, val)
end

function gguf_set_val_bool(ctx, key, val)
    ccall((:gguf_set_val_bool, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, Bool), ctx, key, val)
end

function gguf_set_val_str(ctx, key, val)
    ccall((:gguf_set_val_str, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, Ptr{Cchar}), ctx, key, val)
end

function gguf_set_arr_data(ctx, key, type, data, n)
    ccall((:gguf_set_arr_data, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, gguf_type, Ptr{Cvoid}, Csize_t), ctx, key, type, data, n)
end

function gguf_set_arr_str(ctx, key, data, n)
    ccall((:gguf_set_arr_str, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, Ptr{Ptr{Cchar}}, Csize_t), ctx, key, data, n)
end

function gguf_set_kv(ctx, src)
    ccall((:gguf_set_kv, libllama), Cvoid, (Ptr{gguf_context}, Ptr{gguf_context}), ctx, src)
end

function gguf_add_tensor(ctx, tensor)
    ccall((:gguf_add_tensor, libllama), Cvoid, (Ptr{gguf_context}, Ptr{ggml_tensor}), ctx, tensor)
end

function gguf_set_tensor_type(ctx, name, type)
    ccall((:gguf_set_tensor_type, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, ggml_type), ctx, name, type)
end

function gguf_set_tensor_data(ctx, name, data)
    ccall((:gguf_set_tensor_data, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cchar}, Ptr{Cvoid}), ctx, name, data)
end

function gguf_write_to_file_ptr(ctx, file, only_meta)
    ccall((:gguf_write_to_file_ptr, libllama), Bool, (Ptr{gguf_context}, Ptr{Libc.FILE}, Bool), ctx, file, only_meta)
end

function gguf_write_to_file(ctx, fname, only_meta)
    ccall((:gguf_write_to_file, libllama), Bool, (Ptr{gguf_context}, Ptr{Cchar}, Bool), ctx, fname, only_meta)
end

function gguf_get_meta_size(ctx)
    ccall((:gguf_get_meta_size, libllama), Csize_t, (Ptr{gguf_context},), ctx)
end

function gguf_get_meta_data(ctx, data)
    ccall((:gguf_get_meta_data, libllama), Cvoid, (Ptr{gguf_context}, Ptr{Cvoid}), ctx, data)
end

mutable struct llama_vocab end

mutable struct llama_model end

mutable struct llama_memory_i end

const llama_memory_t = Ptr{llama_memory_i}

const llama_pos = Int32

const llama_token = Int32

const llama_seq_id = Int32

@cenum llama_vocab_type::UInt32 begin
    LLAMA_VOCAB_TYPE_NONE = 0
    LLAMA_VOCAB_TYPE_SPM = 1
    LLAMA_VOCAB_TYPE_BPE = 2
    LLAMA_VOCAB_TYPE_WPM = 3
    LLAMA_VOCAB_TYPE_UGM = 4
    LLAMA_VOCAB_TYPE_RWKV = 5
    LLAMA_VOCAB_TYPE_PLAMO2 = 6
end

@cenum llama_rope_type::Int32 begin
    LLAMA_ROPE_TYPE_NONE = -1
    LLAMA_ROPE_TYPE_NORM = 0
    LLAMA_ROPE_TYPE_NEOX = 2
    LLAMA_ROPE_TYPE_MROPE = 8
    LLAMA_ROPE_TYPE_IMROPE = 40
    LLAMA_ROPE_TYPE_VISION = 24
end

@cenum llama_token_type::UInt32 begin
    LLAMA_TOKEN_TYPE_UNDEFINED = 0
    LLAMA_TOKEN_TYPE_NORMAL = 1
    LLAMA_TOKEN_TYPE_UNKNOWN = 2
    LLAMA_TOKEN_TYPE_CONTROL = 3
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4
    LLAMA_TOKEN_TYPE_UNUSED = 5
    LLAMA_TOKEN_TYPE_BYTE = 6
end

@cenum llama_token_attr::UInt32 begin
    LLAMA_TOKEN_ATTR_UNDEFINED = 0
    LLAMA_TOKEN_ATTR_UNKNOWN = 1
    LLAMA_TOKEN_ATTR_UNUSED = 2
    LLAMA_TOKEN_ATTR_NORMAL = 4
    LLAMA_TOKEN_ATTR_CONTROL = 8
    LLAMA_TOKEN_ATTR_USER_DEFINED = 16
    LLAMA_TOKEN_ATTR_BYTE = 32
    LLAMA_TOKEN_ATTR_NORMALIZED = 64
    LLAMA_TOKEN_ATTR_LSTRIP = 128
    LLAMA_TOKEN_ATTR_RSTRIP = 256
    LLAMA_TOKEN_ATTR_SINGLE_WORD = 512
end

@cenum llama_ftype::UInt32 begin
    LLAMA_FTYPE_ALL_F32 = 0
    LLAMA_FTYPE_MOSTLY_F16 = 1
    LLAMA_FTYPE_MOSTLY_Q4_0 = 2
    LLAMA_FTYPE_MOSTLY_Q4_1 = 3
    LLAMA_FTYPE_MOSTLY_Q8_0 = 7
    LLAMA_FTYPE_MOSTLY_Q5_0 = 8
    LLAMA_FTYPE_MOSTLY_Q5_1 = 9
    LLAMA_FTYPE_MOSTLY_Q2_K = 10
    LLAMA_FTYPE_MOSTLY_Q3_K_S = 11
    LLAMA_FTYPE_MOSTLY_Q3_K_M = 12
    LLAMA_FTYPE_MOSTLY_Q3_K_L = 13
    LLAMA_FTYPE_MOSTLY_Q4_K_S = 14
    LLAMA_FTYPE_MOSTLY_Q4_K_M = 15
    LLAMA_FTYPE_MOSTLY_Q5_K_S = 16
    LLAMA_FTYPE_MOSTLY_Q5_K_M = 17
    LLAMA_FTYPE_MOSTLY_Q6_K = 18
    LLAMA_FTYPE_MOSTLY_IQ2_XXS = 19
    LLAMA_FTYPE_MOSTLY_IQ2_XS = 20
    LLAMA_FTYPE_MOSTLY_Q2_K_S = 21
    LLAMA_FTYPE_MOSTLY_IQ3_XS = 22
    LLAMA_FTYPE_MOSTLY_IQ3_XXS = 23
    LLAMA_FTYPE_MOSTLY_IQ1_S = 24
    LLAMA_FTYPE_MOSTLY_IQ4_NL = 25
    LLAMA_FTYPE_MOSTLY_IQ3_S = 26
    LLAMA_FTYPE_MOSTLY_IQ3_M = 27
    LLAMA_FTYPE_MOSTLY_IQ2_S = 28
    LLAMA_FTYPE_MOSTLY_IQ2_M = 29
    LLAMA_FTYPE_MOSTLY_IQ4_XS = 30
    LLAMA_FTYPE_MOSTLY_IQ1_M = 31
    LLAMA_FTYPE_MOSTLY_BF16 = 32
    LLAMA_FTYPE_MOSTLY_TQ1_0 = 36
    LLAMA_FTYPE_MOSTLY_TQ2_0 = 37
    LLAMA_FTYPE_MOSTLY_MXFP4_MOE = 38
    LLAMA_FTYPE_MOSTLY_NVFP4 = 39
    LLAMA_FTYPE_MOSTLY_Q1_0 = 40
    LLAMA_FTYPE_MOSTLY_Q2_0 = 41
    LLAMA_FTYPE_GUESSED = 1024
end

function llama_ftype_name(ftype)
    ccall((:llama_ftype_name, libllama), Ptr{Cchar}, (llama_ftype,), ftype)
end

@cenum llama_rope_scaling_type::Int32 begin
    LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1
    LLAMA_ROPE_SCALING_TYPE_NONE = 0
    LLAMA_ROPE_SCALING_TYPE_LINEAR = 1
    LLAMA_ROPE_SCALING_TYPE_YARN = 2
    LLAMA_ROPE_SCALING_TYPE_LONGROPE = 3
    LLAMA_ROPE_SCALING_TYPE_MAX_VALUE = 3
end

@cenum llama_pooling_type::Int32 begin
    LLAMA_POOLING_TYPE_UNSPECIFIED = -1
    LLAMA_POOLING_TYPE_NONE = 0
    LLAMA_POOLING_TYPE_MEAN = 1
    LLAMA_POOLING_TYPE_CLS = 2
    LLAMA_POOLING_TYPE_LAST = 3
    LLAMA_POOLING_TYPE_RANK = 4
end

@cenum llama_attention_type::Int32 begin
    LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1
    LLAMA_ATTENTION_TYPE_CAUSAL = 0
    LLAMA_ATTENTION_TYPE_NON_CAUSAL = 1
end

@cenum llama_flash_attn_type::Int32 begin
    LLAMA_FLASH_ATTN_TYPE_AUTO = -1
    LLAMA_FLASH_ATTN_TYPE_DISABLED = 0
    LLAMA_FLASH_ATTN_TYPE_ENABLED = 1
end

function llama_flash_attn_type_name(flash_attn_type)
    ccall((:llama_flash_attn_type_name, libllama), Ptr{Cchar}, (llama_flash_attn_type,), flash_attn_type)
end

@cenum llama_split_mode::UInt32 begin
    LLAMA_SPLIT_MODE_NONE = 0
    LLAMA_SPLIT_MODE_LAYER = 1
    LLAMA_SPLIT_MODE_ROW = 2
    LLAMA_SPLIT_MODE_TENSOR = 3
end

@cenum llama_load_mode::Int32 begin
    LLAMA_LOAD_MODE_AUTO = -1
    LLAMA_LOAD_MODE_NONE = 0
    LLAMA_LOAD_MODE_MMAP = 1
    LLAMA_LOAD_MODE_MLOCK = 2
    LLAMA_LOAD_MODE_MMAP_MLOCK = 3
    LLAMA_LOAD_MODE_DIRECT_IO = 4
end

function llama_load_mode_name(load_mode)
    ccall((:llama_load_mode_name, libllama), Ptr{Cchar}, (llama_load_mode,), load_mode)
end

function llama_load_mode_from_str(str)
    ccall((:llama_load_mode_from_str, libllama), llama_load_mode, (Ptr{Cchar},), str)
end

@cenum llama_context_type::UInt32 begin
    LLAMA_CONTEXT_TYPE_DEFAULT = 0
    LLAMA_CONTEXT_TYPE_MTP = 1
end

struct llama_token_data
    id::llama_token
    logit::Cfloat
    p::Cfloat
end

struct llama_token_data_array
    data::Ptr{llama_token_data}
    size::Csize_t
    selected::Int64
    sorted::Bool
end

# typedef bool ( * llama_progress_callback ) ( float progress , void * user_data )
const llama_progress_callback = Ptr{Cvoid}

struct llama_batch
    n_tokens::Int32
    token::Ptr{llama_token}
    embd::Ptr{Cfloat}
    pos::Ptr{llama_pos}
    n_seq_id::Ptr{Int32}
    seq_id::Ptr{Ptr{llama_seq_id}}
    logits::Ptr{Int8}
end

@cenum llama_model_kv_override_type::UInt32 begin
    LLAMA_KV_OVERRIDE_TYPE_INT = 0
    LLAMA_KV_OVERRIDE_TYPE_FLOAT = 1
    LLAMA_KV_OVERRIDE_TYPE_BOOL = 2
    LLAMA_KV_OVERRIDE_TYPE_STR = 3
end

@cenum llama_model_meta_key::UInt32 begin
    LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE = 0
    LLAMA_MODEL_META_KEY_SAMPLING_TOP_K = 1
    LLAMA_MODEL_META_KEY_SAMPLING_TOP_P = 2
    LLAMA_MODEL_META_KEY_SAMPLING_MIN_P = 3
    LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY = 4
    LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD = 5
    LLAMA_MODEL_META_KEY_SAMPLING_TEMP = 6
    LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N = 7
    LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT = 8
    LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT = 9
    LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU = 10
    LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA = 11
end

struct llama_model_kv_override
    data::NTuple{264, UInt8}
end

function Base.getproperty(x::Ptr{llama_model_kv_override}, f::Symbol)
    f === :tag && return Ptr{llama_model_kv_override_type}(x + 0)
    f === :key && return Ptr{NTuple{128, Cchar}}(x + 4)
    f === :val_i64 && return Ptr{Int64}(x + 136)
    f === :val_f64 && return Ptr{Cdouble}(x + 136)
    f === :val_bool && return Ptr{Bool}(x + 136)
    f === :val_str && return Ptr{NTuple{128, Cchar}}(x + 136)
    return getfield(x, f)
end

function Base.getproperty(x::llama_model_kv_override, f::Symbol)
    r = Ref{llama_model_kv_override}(x)
    ptr = Base.unsafe_convert(Ptr{llama_model_kv_override}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{llama_model_kv_override}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::llama_model_kv_override, private::Bool = false)
    (:tag, :key, :val_i64, :val_f64, :val_bool, :val_str, if private
            fieldnames(typeof(x))
        else
            ()
        end...)
end

struct llama_model_tensor_buft_override
    pattern::Ptr{Cchar}
    buft::ggml_backend_buffer_type_t
end

struct llama_model_params
    devices::Ptr{ggml_backend_dev_t}
    tensor_buft_overrides::Ptr{llama_model_tensor_buft_override}
    n_gpu_layers::Int32
    split_mode::llama_split_mode
    load_mode::llama_load_mode
    main_gpu::Int32
    tensor_split::Ptr{Cfloat}
    progress_callback::llama_progress_callback
    progress_callback_user_data::Ptr{Cvoid}
    kv_overrides::Ptr{llama_model_kv_override}
    vocab_only::Bool
    check_tensors::Bool
    use_extra_bufts::Bool
    no_host::Bool
    no_alloc::Bool
    load_mtp::Bool
end

struct llama_sampler_i
    name::Ptr{Cvoid}
    accept::Ptr{Cvoid}
    apply::Ptr{Cvoid}
    reset::Ptr{Cvoid}
    clone::Ptr{Cvoid}
    free::Ptr{Cvoid}
    backend_init::Ptr{Cvoid}
    backend_accept::Ptr{Cvoid}
    backend_apply::Ptr{Cvoid}
    backend_set_input::Ptr{Cvoid}
    backend_reset::Ptr{Cvoid}
    copy_state::Ptr{Cvoid}
end

const llama_sampler_context_t = Ptr{Cvoid}

struct llama_sampler
    iface::Ptr{llama_sampler_i}
    ctx::llama_sampler_context_t
end

struct llama_sampler_seq_config
    seq_id::llama_seq_id
    sampler::Ptr{llama_sampler}
end

mutable struct llama_context end

struct llama_context_params
    n_ctx::UInt32
    n_batch::UInt32
    n_ubatch::UInt32
    n_seq_max::UInt32
    n_rs_seq::UInt32
    n_outputs_max::UInt32
    n_outputs_max_per_seq::UInt32
    n_threads::Int32
    n_threads_batch::Int32
    ctx_type::llama_context_type
    rope_scaling_type::llama_rope_scaling_type
    pooling_type::llama_pooling_type
    attention_type::llama_attention_type
    flash_attn_type::llama_flash_attn_type
    rope_freq_base::Cfloat
    rope_freq_scale::Cfloat
    yarn_ext_factor::Cfloat
    yarn_attn_factor::Cfloat
    yarn_beta_fast::Cfloat
    yarn_beta_slow::Cfloat
    yarn_orig_ctx::UInt32
    defrag_thold::Cfloat
    cb_eval::ggml_backend_sched_eval_callback
    cb_eval_user_data::Ptr{Cvoid}
    type_k::ggml_type
    type_v::ggml_type
    abort_callback::ggml_abort_callback
    abort_callback_data::Ptr{Cvoid}
    embeddings::Bool
    offload_kqv::Bool
    no_perf::Bool
    op_offload::Bool
    swa_full::Bool
    kv_unified::Bool
    samplers::Ptr{llama_sampler_seq_config}
    n_samplers::Csize_t
    ctx_other::Ptr{llama_context}
end

struct llama_model_tensor_override
    pattern::Ptr{Cchar}
    type::ggml_type
end

struct llama_model_imatrix_data
    name::Ptr{Cchar}
    data::Ptr{Cfloat}
    size::Csize_t
end

struct llama_model_quantize_params
    nthread::Int32
    ftype::llama_ftype
    output_tensor_type::ggml_type
    token_embedding_type::ggml_type
    allow_requantize::Bool
    quantize_output_tensor::Bool
    only_copy::Bool
    pure::Bool
    keep_split::Bool
    dry_run::Bool
    imatrix::Ptr{llama_model_imatrix_data}
    kv_overrides::Ptr{llama_model_kv_override}
    tt_overrides::Ptr{llama_model_tensor_override}
    prune_layers::Ptr{Int32}
end

struct llama_logit_bias
    token::llama_token
    bias::Cfloat
end

struct llama_sampler_chain_params
    no_perf::Bool
end

struct llama_chat_message
    role::Ptr{Cchar}
    content::Ptr{Cchar}
end

mutable struct llama_adapter_lora end

function llama_version()
    ccall((:llama_version, libllama), Ptr{Cchar}, ())
end

function llama_model_default_params()
    ccall((:llama_model_default_params, libllama), llama_model_params, ())
end

function llama_context_default_params()
    ccall((:llama_context_default_params, libllama), llama_context_params, ())
end

function llama_sampler_chain_default_params()
    ccall((:llama_sampler_chain_default_params, libllama), llama_sampler_chain_params, ())
end

function llama_model_quantize_default_params()
    ccall((:llama_model_quantize_default_params, libllama), llama_model_quantize_params, ())
end

function llama_backend_init()
    ccall((:llama_backend_init, libllama), Cvoid, ())
end

function llama_backend_free()
    ccall((:llama_backend_free, libllama), Cvoid, ())
end

function llama_numa_init(numa)
    ccall((:llama_numa_init, libllama), Cvoid, (ggml_numa_strategy,), numa)
end

function llama_attach_threadpool(ctx, threadpool, threadpool_batch)
    ccall((:llama_attach_threadpool, libllama), Cvoid, (Ptr{llama_context}, ggml_threadpool_t, ggml_threadpool_t), ctx, threadpool, threadpool_batch)
end

function llama_detach_threadpool(ctx)
    ccall((:llama_detach_threadpool, libllama), Cvoid, (Ptr{llama_context},), ctx)
end

# typedef void ( * llama_model_set_tensor_data_t ) ( struct ggml_tensor * tensor , void * userdata )
const llama_model_set_tensor_data_t = Ptr{Cvoid}

function llama_model_init_from_user(metadata, set_tensor_data, set_tensor_data_ud, params)
    ccall((:llama_model_init_from_user, libllama), Ptr{llama_model}, (Ptr{gguf_context}, llama_model_set_tensor_data_t, Ptr{Cvoid}, llama_model_params), metadata, set_tensor_data, set_tensor_data_ud, params)
end

function llama_load_model_from_file(path_model, params)
    ccall((:llama_load_model_from_file, libllama), Ptr{llama_model}, (Ptr{Cchar}, llama_model_params), path_model, params)
end

function llama_model_load_from_file(path_model, params)
    ccall((:llama_model_load_from_file, libllama), Ptr{llama_model}, (Ptr{Cchar}, llama_model_params), path_model, params)
end

function llama_model_load_from_file_ptr(file, params)
    ccall((:llama_model_load_from_file_ptr, libllama), Ptr{llama_model}, (Ptr{Libc.FILE}, llama_model_params), file, params)
end

function llama_model_load_from_splits(paths, n_paths, params)
    ccall((:llama_model_load_from_splits, libllama), Ptr{llama_model}, (Ptr{Ptr{Cchar}}, Csize_t, llama_model_params), paths, n_paths, params)
end

function llama_model_save_to_file(model, path_model)
    ccall((:llama_model_save_to_file, libllama), Cvoid, (Ptr{llama_model}, Ptr{Cchar}), model, path_model)
end

function llama_free_model(model)
    ccall((:llama_free_model, libllama), Cvoid, (Ptr{llama_model},), model)
end

function llama_model_free(model)
    ccall((:llama_model_free, libllama), Cvoid, (Ptr{llama_model},), model)
end

function llama_init_from_model(model, params)
    ccall((:llama_init_from_model, libllama), Ptr{llama_context}, (Ptr{llama_model}, llama_context_params), model, params)
end

function llama_new_context_with_model(model, params)
    ccall((:llama_new_context_with_model, libllama), Ptr{llama_context}, (Ptr{llama_model}, llama_context_params), model, params)
end

function llama_free(ctx)
    ccall((:llama_free, libllama), Cvoid, (Ptr{llama_context},), ctx)
end

function llama_time_us()
    ccall((:llama_time_us, libllama), Int64, ())
end

function llama_max_devices()
    ccall((:llama_max_devices, libllama), Csize_t, ())
end

function llama_max_parallel_sequences()
    ccall((:llama_max_parallel_sequences, libllama), Csize_t, ())
end

function llama_max_tensor_buft_overrides()
    ccall((:llama_max_tensor_buft_overrides, libllama), Csize_t, ())
end

function llama_supports_mmap()
    ccall((:llama_supports_mmap, libllama), Bool, ())
end

function llama_supports_mlock()
    ccall((:llama_supports_mlock, libllama), Bool, ())
end

function llama_supports_gpu_offload()
    ccall((:llama_supports_gpu_offload, libllama), Bool, ())
end

function llama_supports_rpc()
    ccall((:llama_supports_rpc, libllama), Bool, ())
end

function llama_n_ctx(ctx)
    ccall((:llama_n_ctx, libllama), UInt32, (Ptr{llama_context},), ctx)
end

function llama_n_ctx_seq(ctx)
    ccall((:llama_n_ctx_seq, libllama), UInt32, (Ptr{llama_context},), ctx)
end

function llama_n_batch(ctx)
    ccall((:llama_n_batch, libllama), UInt32, (Ptr{llama_context},), ctx)
end

function llama_n_ubatch(ctx)
    ccall((:llama_n_ubatch, libllama), UInt32, (Ptr{llama_context},), ctx)
end

function llama_n_seq_max(ctx)
    ccall((:llama_n_seq_max, libllama), UInt32, (Ptr{llama_context},), ctx)
end

function llama_n_rs_seq(ctx)
    ccall((:llama_n_rs_seq, libllama), UInt32, (Ptr{llama_context},), ctx)
end

function llama_n_ctx_train(model)
    ccall((:llama_n_ctx_train, libllama), Int32, (Ptr{llama_model},), model)
end

function llama_n_embd(model)
    ccall((:llama_n_embd, libllama), Int32, (Ptr{llama_model},), model)
end

function llama_n_layer(model)
    ccall((:llama_n_layer, libllama), Int32, (Ptr{llama_model},), model)
end

function llama_n_head(model)
    ccall((:llama_n_head, libllama), Int32, (Ptr{llama_model},), model)
end

function llama_n_vocab(vocab)
    ccall((:llama_n_vocab, libllama), Int32, (Ptr{llama_vocab},), vocab)
end

function llama_get_model(ctx)
    ccall((:llama_get_model, libllama), Ptr{llama_model}, (Ptr{llama_context},), ctx)
end

function llama_get_memory(ctx)
    ccall((:llama_get_memory, libllama), llama_memory_t, (Ptr{llama_context},), ctx)
end

function llama_pooling_type(ctx)
    ccall((:llama_pooling_type, libllama), llama_pooling_type, (Ptr{llama_context},), ctx)
end

function llama_model_get_vocab(model)
    ccall((:llama_model_get_vocab, libllama), Ptr{llama_vocab}, (Ptr{llama_model},), model)
end

function llama_model_rope_type(model)
    ccall((:llama_model_rope_type, libllama), llama_rope_type, (Ptr{llama_model},), model)
end

function llama_model_n_ctx_train(model)
    ccall((:llama_model_n_ctx_train, libllama), Int32, (Ptr{llama_model},), model)
end

function llama_model_n_embd(model)
    ccall((:llama_model_n_embd, libllama), Int32, (Ptr{llama_model},), model)
end

function llama_model_n_embd_inp(model)
    ccall((:llama_model_n_embd_inp, libllama), Int32, (Ptr{llama_model},), model)
end

function llama_model_n_embd_out(model)
    ccall((:llama_model_n_embd_out, libllama), Int32, (Ptr{llama_model},), model)
end

function llama_model_n_layer(model)
    ccall((:llama_model_n_layer, libllama), Int32, (Ptr{llama_model},), model)
end

function llama_model_n_layer_nextn(model)
    ccall((:llama_model_n_layer_nextn, libllama), Int32, (Ptr{llama_model},), model)
end

function llama_model_n_head(model)
    ccall((:llama_model_n_head, libllama), Int32, (Ptr{llama_model},), model)
end

function llama_model_n_head_kv(model)
    ccall((:llama_model_n_head_kv, libllama), Int32, (Ptr{llama_model},), model)
end

function llama_model_n_swa(model)
    ccall((:llama_model_n_swa, libllama), Int32, (Ptr{llama_model},), model)
end

function llama_model_rope_freq_scale_train(model)
    ccall((:llama_model_rope_freq_scale_train, libllama), Cfloat, (Ptr{llama_model},), model)
end

function llama_model_n_cls_out(model)
    ccall((:llama_model_n_cls_out, libllama), UInt32, (Ptr{llama_model},), model)
end

function llama_model_cls_label(model, i)
    ccall((:llama_model_cls_label, libllama), Ptr{Cchar}, (Ptr{llama_model}, UInt32), model, i)
end

function llama_vocab_type(vocab)
    ccall((:llama_vocab_type, libllama), llama_vocab_type, (Ptr{llama_vocab},), vocab)
end

function llama_vocab_n_tokens(vocab)
    ccall((:llama_vocab_n_tokens, libllama), Int32, (Ptr{llama_vocab},), vocab)
end

function llama_model_meta_val_str(model, key, buf, buf_size)
    ccall((:llama_model_meta_val_str, libllama), Int32, (Ptr{llama_model}, Ptr{Cchar}, Ptr{Cchar}, Csize_t), model, key, buf, buf_size)
end

function llama_model_meta_count(model)
    ccall((:llama_model_meta_count, libllama), Int32, (Ptr{llama_model},), model)
end

function llama_model_meta_key_str(key)
    ccall((:llama_model_meta_key_str, libllama), Ptr{Cchar}, (llama_model_meta_key,), key)
end

function llama_model_meta_key_by_index(model, i, buf, buf_size)
    ccall((:llama_model_meta_key_by_index, libllama), Int32, (Ptr{llama_model}, Int32, Ptr{Cchar}, Csize_t), model, i, buf, buf_size)
end

function llama_model_meta_val_str_by_index(model, i, buf, buf_size)
    ccall((:llama_model_meta_val_str_by_index, libllama), Int32, (Ptr{llama_model}, Int32, Ptr{Cchar}, Csize_t), model, i, buf, buf_size)
end

function llama_model_desc(model, buf, buf_size)
    ccall((:llama_model_desc, libllama), Int32, (Ptr{llama_model}, Ptr{Cchar}, Csize_t), model, buf, buf_size)
end

function llama_model_ftype(model)
    ccall((:llama_model_ftype, libllama), llama_ftype, (Ptr{llama_model},), model)
end

function llama_model_size(model)
    ccall((:llama_model_size, libllama), UInt64, (Ptr{llama_model},), model)
end

function llama_model_chat_template(model, name)
    ccall((:llama_model_chat_template, libllama), Ptr{Cchar}, (Ptr{llama_model}, Ptr{Cchar}), model, name)
end

function llama_model_n_params(model)
    ccall((:llama_model_n_params, libllama), UInt64, (Ptr{llama_model},), model)
end

function llama_model_has_encoder(model)
    ccall((:llama_model_has_encoder, libllama), Bool, (Ptr{llama_model},), model)
end

function llama_model_has_decoder(model)
    ccall((:llama_model_has_decoder, libllama), Bool, (Ptr{llama_model},), model)
end

function llama_model_decoder_start_token(model)
    ccall((:llama_model_decoder_start_token, libllama), llama_token, (Ptr{llama_model},), model)
end

function llama_model_is_recurrent(model)
    ccall((:llama_model_is_recurrent, libllama), Bool, (Ptr{llama_model},), model)
end

function llama_model_is_hybrid(model)
    ccall((:llama_model_is_hybrid, libllama), Bool, (Ptr{llama_model},), model)
end

function llama_model_is_diffusion(model)
    ccall((:llama_model_is_diffusion, libllama), Bool, (Ptr{llama_model},), model)
end

function llama_model_quantize(fname_inp, fname_out, params)
    ccall((:llama_model_quantize, libllama), UInt32, (Ptr{Cchar}, Ptr{Cchar}, Ptr{llama_model_quantize_params}), fname_inp, fname_out, params)
end

function llama_adapter_lora_init(model, path_lora)
    ccall((:llama_adapter_lora_init, libllama), Ptr{llama_adapter_lora}, (Ptr{llama_model}, Ptr{Cchar}), model, path_lora)
end

function llama_adapter_meta_val_str(adapter, key, buf, buf_size)
    ccall((:llama_adapter_meta_val_str, libllama), Int32, (Ptr{llama_adapter_lora}, Ptr{Cchar}, Ptr{Cchar}, Csize_t), adapter, key, buf, buf_size)
end

function llama_adapter_meta_count(adapter)
    ccall((:llama_adapter_meta_count, libllama), Int32, (Ptr{llama_adapter_lora},), adapter)
end

function llama_adapter_meta_key_by_index(adapter, i, buf, buf_size)
    ccall((:llama_adapter_meta_key_by_index, libllama), Int32, (Ptr{llama_adapter_lora}, Int32, Ptr{Cchar}, Csize_t), adapter, i, buf, buf_size)
end

function llama_adapter_meta_val_str_by_index(adapter, i, buf, buf_size)
    ccall((:llama_adapter_meta_val_str_by_index, libllama), Int32, (Ptr{llama_adapter_lora}, Int32, Ptr{Cchar}, Csize_t), adapter, i, buf, buf_size)
end

function llama_adapter_lora_free(adapter)
    ccall((:llama_adapter_lora_free, libllama), Cvoid, (Ptr{llama_adapter_lora},), adapter)
end

function llama_adapter_get_alora_n_invocation_tokens(adapter)
    ccall((:llama_adapter_get_alora_n_invocation_tokens, libllama), UInt64, (Ptr{llama_adapter_lora},), adapter)
end

function llama_adapter_get_alora_invocation_tokens(adapter)
    ccall((:llama_adapter_get_alora_invocation_tokens, libllama), Ptr{llama_token}, (Ptr{llama_adapter_lora},), adapter)
end

function llama_set_adapters_lora(ctx, adapters, n_adapters, scales)
    ccall((:llama_set_adapters_lora, libllama), Int32, (Ptr{llama_context}, Ptr{Ptr{llama_adapter_lora}}, Csize_t, Ptr{Cfloat}), ctx, adapters, n_adapters, scales)
end

function llama_set_adapter_cvec(ctx, data, len, n_embd, il_start, il_end)
    ccall((:llama_set_adapter_cvec, libllama), Int32, (Ptr{llama_context}, Ptr{Cfloat}, Csize_t, Int32, Int32, Int32), ctx, data, len, n_embd, il_start, il_end)
end

function llama_memory_clear(mem, data)
    ccall((:llama_memory_clear, libllama), Cvoid, (llama_memory_t, Bool), mem, data)
end

function llama_memory_seq_rm(mem, seq_id, p0, p1)
    ccall((:llama_memory_seq_rm, libllama), Bool, (llama_memory_t, llama_seq_id, llama_pos, llama_pos), mem, seq_id, p0, p1)
end

function llama_memory_seq_cp(mem, seq_id_src, seq_id_dst, p0, p1)
    ccall((:llama_memory_seq_cp, libllama), Cvoid, (llama_memory_t, llama_seq_id, llama_seq_id, llama_pos, llama_pos), mem, seq_id_src, seq_id_dst, p0, p1)
end

function llama_memory_seq_keep(mem, seq_id)
    ccall((:llama_memory_seq_keep, libllama), Cvoid, (llama_memory_t, llama_seq_id), mem, seq_id)
end

function llama_memory_seq_add(mem, seq_id, p0, p1, delta)
    ccall((:llama_memory_seq_add, libllama), Cvoid, (llama_memory_t, llama_seq_id, llama_pos, llama_pos, llama_pos), mem, seq_id, p0, p1, delta)
end

function llama_memory_seq_div(mem, seq_id, p0, p1, d)
    ccall((:llama_memory_seq_div, libllama), Cvoid, (llama_memory_t, llama_seq_id, llama_pos, llama_pos, Cint), mem, seq_id, p0, p1, d)
end

function llama_memory_seq_pos_min(mem, seq_id)
    ccall((:llama_memory_seq_pos_min, libllama), llama_pos, (llama_memory_t, llama_seq_id), mem, seq_id)
end

function llama_memory_seq_pos_max(mem, seq_id)
    ccall((:llama_memory_seq_pos_max, libllama), llama_pos, (llama_memory_t, llama_seq_id), mem, seq_id)
end

function llama_memory_can_shift(mem)
    ccall((:llama_memory_can_shift, libllama), Bool, (llama_memory_t,), mem)
end

function llama_state_get_size(ctx)
    ccall((:llama_state_get_size, libllama), Csize_t, (Ptr{llama_context},), ctx)
end

function llama_get_state_size(ctx)
    ccall((:llama_get_state_size, libllama), Csize_t, (Ptr{llama_context},), ctx)
end

function llama_state_get_data(ctx, dst, size)
    ccall((:llama_state_get_data, libllama), Csize_t, (Ptr{llama_context}, Ptr{UInt8}, Csize_t), ctx, dst, size)
end

function llama_copy_state_data(ctx, dst)
    ccall((:llama_copy_state_data, libllama), Csize_t, (Ptr{llama_context}, Ptr{UInt8}), ctx, dst)
end

function llama_state_set_data(ctx, src, size)
    ccall((:llama_state_set_data, libllama), Csize_t, (Ptr{llama_context}, Ptr{UInt8}, Csize_t), ctx, src, size)
end

function llama_set_state_data(ctx, src)
    ccall((:llama_set_state_data, libllama), Csize_t, (Ptr{llama_context}, Ptr{UInt8}), ctx, src)
end

function llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out)
    ccall((:llama_state_load_file, libllama), Bool, (Ptr{llama_context}, Ptr{Cchar}, Ptr{llama_token}, Csize_t, Ptr{Csize_t}), ctx, path_session, tokens_out, n_token_capacity, n_token_count_out)
end

function llama_load_session_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out)
    ccall((:llama_load_session_file, libllama), Bool, (Ptr{llama_context}, Ptr{Cchar}, Ptr{llama_token}, Csize_t, Ptr{Csize_t}), ctx, path_session, tokens_out, n_token_capacity, n_token_count_out)
end

function llama_state_save_file(ctx, path_session, tokens, n_token_count)
    ccall((:llama_state_save_file, libllama), Bool, (Ptr{llama_context}, Ptr{Cchar}, Ptr{llama_token}, Csize_t), ctx, path_session, tokens, n_token_count)
end

function llama_save_session_file(ctx, path_session, tokens, n_token_count)
    ccall((:llama_save_session_file, libllama), Bool, (Ptr{llama_context}, Ptr{Cchar}, Ptr{llama_token}, Csize_t), ctx, path_session, tokens, n_token_count)
end

function llama_state_seq_get_size(ctx, seq_id)
    ccall((:llama_state_seq_get_size, libllama), Csize_t, (Ptr{llama_context}, llama_seq_id), ctx, seq_id)
end

function llama_state_seq_get_data(ctx, dst, size, seq_id)
    ccall((:llama_state_seq_get_data, libllama), Csize_t, (Ptr{llama_context}, Ptr{UInt8}, Csize_t, llama_seq_id), ctx, dst, size, seq_id)
end

function llama_state_seq_set_data(ctx, src, size, dest_seq_id)
    ccall((:llama_state_seq_set_data, libllama), Csize_t, (Ptr{llama_context}, Ptr{UInt8}, Csize_t, llama_seq_id), ctx, src, size, dest_seq_id)
end

function llama_state_seq_save_file(ctx, filepath, seq_id, tokens, n_token_count)
    ccall((:llama_state_seq_save_file, libllama), Csize_t, (Ptr{llama_context}, Ptr{Cchar}, llama_seq_id, Ptr{llama_token}, Csize_t), ctx, filepath, seq_id, tokens, n_token_count)
end

function llama_state_seq_load_file(ctx, filepath, dest_seq_id, tokens_out, n_token_capacity, n_token_count_out)
    ccall((:llama_state_seq_load_file, libllama), Csize_t, (Ptr{llama_context}, Ptr{Cchar}, llama_seq_id, Ptr{llama_token}, Csize_t, Ptr{Csize_t}), ctx, filepath, dest_seq_id, tokens_out, n_token_capacity, n_token_count_out)
end

const llama_state_seq_flags = UInt32

function llama_state_seq_get_size_ext(ctx, seq_id, flags)
    ccall((:llama_state_seq_get_size_ext, libllama), Csize_t, (Ptr{llama_context}, llama_seq_id, llama_state_seq_flags), ctx, seq_id, flags)
end

function llama_state_seq_get_data_ext(ctx, dst, size, seq_id, flags)
    ccall((:llama_state_seq_get_data_ext, libllama), Csize_t, (Ptr{llama_context}, Ptr{UInt8}, Csize_t, llama_seq_id, llama_state_seq_flags), ctx, dst, size, seq_id, flags)
end

function llama_state_seq_set_data_ext(ctx, src, size, dest_seq_id, flags)
    ccall((:llama_state_seq_set_data_ext, libllama), Csize_t, (Ptr{llama_context}, Ptr{UInt8}, Csize_t, llama_seq_id, llama_state_seq_flags), ctx, src, size, dest_seq_id, flags)
end

function llama_batch_get_one(tokens, n_tokens)
    ccall((:llama_batch_get_one, libllama), llama_batch, (Ptr{llama_token}, Int32), tokens, n_tokens)
end

function llama_batch_init(n_tokens, embd, n_seq_max)
    ccall((:llama_batch_init, libllama), llama_batch, (Int32, Int32, Int32), n_tokens, embd, n_seq_max)
end

function llama_batch_free(batch)
    ccall((:llama_batch_free, libllama), Cvoid, (llama_batch,), batch)
end

function llama_encode(ctx, batch)
    ccall((:llama_encode, libllama), Int32, (Ptr{llama_context}, llama_batch), ctx, batch)
end

function llama_decode(ctx, batch)
    ccall((:llama_decode, libllama), Int32, (Ptr{llama_context}, llama_batch), ctx, batch)
end

function llama_set_n_threads(ctx, n_threads, n_threads_batch)
    ccall((:llama_set_n_threads, libllama), Cvoid, (Ptr{llama_context}, Int32, Int32), ctx, n_threads, n_threads_batch)
end

function llama_n_threads(ctx)
    ccall((:llama_n_threads, libllama), Int32, (Ptr{llama_context},), ctx)
end

function llama_n_threads_batch(ctx)
    ccall((:llama_n_threads_batch, libllama), Int32, (Ptr{llama_context},), ctx)
end

function llama_set_embeddings(ctx, embeddings)
    ccall((:llama_set_embeddings, libllama), Cvoid, (Ptr{llama_context}, Bool), ctx, embeddings)
end

function llama_set_causal_attn(ctx, causal_attn)
    ccall((:llama_set_causal_attn, libllama), Cvoid, (Ptr{llama_context}, Bool), ctx, causal_attn)
end

function llama_set_warmup(ctx, warmup)
    ccall((:llama_set_warmup, libllama), Cvoid, (Ptr{llama_context}, Bool), ctx, warmup)
end

function llama_set_abort_callback(ctx, abort_callback, abort_callback_data)
    ccall((:llama_set_abort_callback, libllama), Cvoid, (Ptr{llama_context}, ggml_abort_callback, Ptr{Cvoid}), ctx, abort_callback, abort_callback_data)
end

function llama_synchronize(ctx)
    ccall((:llama_synchronize, libllama), Cvoid, (Ptr{llama_context},), ctx)
end

function llama_get_logits(ctx)
    ccall((:llama_get_logits, libllama), Ptr{Cfloat}, (Ptr{llama_context},), ctx)
end

function llama_get_logits_ith(ctx, i)
    ccall((:llama_get_logits_ith, libllama), Ptr{Cfloat}, (Ptr{llama_context}, Int32), ctx, i)
end

function llama_get_embeddings(ctx)
    ccall((:llama_get_embeddings, libllama), Ptr{Cfloat}, (Ptr{llama_context},), ctx)
end

function llama_get_embeddings_ith(ctx, i)
    ccall((:llama_get_embeddings_ith, libllama), Ptr{Cfloat}, (Ptr{llama_context}, Int32), ctx, i)
end

function llama_get_embeddings_seq(ctx, seq_id)
    ccall((:llama_get_embeddings_seq, libllama), Ptr{Cfloat}, (Ptr{llama_context}, llama_seq_id), ctx, seq_id)
end

function llama_get_sampled_token_ith(ctx, i)
    ccall((:llama_get_sampled_token_ith, libllama), llama_token, (Ptr{llama_context}, Int32), ctx, i)
end

function llama_get_sampled_probs_ith(ctx, i)
    ccall((:llama_get_sampled_probs_ith, libllama), Ptr{Cfloat}, (Ptr{llama_context}, Int32), ctx, i)
end

function llama_get_sampled_probs_count_ith(ctx, i)
    ccall((:llama_get_sampled_probs_count_ith, libllama), UInt32, (Ptr{llama_context}, Int32), ctx, i)
end

function llama_get_sampled_logits_ith(ctx, i)
    ccall((:llama_get_sampled_logits_ith, libllama), Ptr{Cfloat}, (Ptr{llama_context}, Int32), ctx, i)
end

function llama_get_sampled_logits_count_ith(ctx, i)
    ccall((:llama_get_sampled_logits_count_ith, libllama), UInt32, (Ptr{llama_context}, Int32), ctx, i)
end

function llama_get_sampled_candidates_ith(ctx, i)
    ccall((:llama_get_sampled_candidates_ith, libllama), Ptr{llama_token}, (Ptr{llama_context}, Int32), ctx, i)
end

function llama_get_sampled_candidates_count_ith(ctx, i)
    ccall((:llama_get_sampled_candidates_count_ith, libllama), UInt32, (Ptr{llama_context}, Int32), ctx, i)
end

function llama_vocab_get_text(vocab, token)
    ccall((:llama_vocab_get_text, libllama), Ptr{Cchar}, (Ptr{llama_vocab}, llama_token), vocab, token)
end

function llama_vocab_get_score(vocab, token)
    ccall((:llama_vocab_get_score, libllama), Cfloat, (Ptr{llama_vocab}, llama_token), vocab, token)
end

function llama_vocab_get_attr(vocab, token)
    ccall((:llama_vocab_get_attr, libllama), llama_token_attr, (Ptr{llama_vocab}, llama_token), vocab, token)
end

function llama_vocab_is_eog(vocab, token)
    ccall((:llama_vocab_is_eog, libllama), Bool, (Ptr{llama_vocab}, llama_token), vocab, token)
end

function llama_vocab_is_control(vocab, token)
    ccall((:llama_vocab_is_control, libllama), Bool, (Ptr{llama_vocab}, llama_token), vocab, token)
end

function llama_vocab_bos(vocab)
    ccall((:llama_vocab_bos, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_vocab_eos(vocab)
    ccall((:llama_vocab_eos, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_vocab_eot(vocab)
    ccall((:llama_vocab_eot, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_vocab_sep(vocab)
    ccall((:llama_vocab_sep, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_vocab_nl(vocab)
    ccall((:llama_vocab_nl, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_vocab_pad(vocab)
    ccall((:llama_vocab_pad, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_vocab_mask(vocab)
    ccall((:llama_vocab_mask, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_vocab_get_add_bos(vocab)
    ccall((:llama_vocab_get_add_bos, libllama), Bool, (Ptr{llama_vocab},), vocab)
end

function llama_vocab_get_add_eos(vocab)
    ccall((:llama_vocab_get_add_eos, libllama), Bool, (Ptr{llama_vocab},), vocab)
end

function llama_vocab_get_add_sep(vocab)
    ccall((:llama_vocab_get_add_sep, libllama), Bool, (Ptr{llama_vocab},), vocab)
end

function llama_vocab_get_suppress_tokens(vocab, n_suppress_tokens)
    ccall((:llama_vocab_get_suppress_tokens, libllama), Ptr{llama_token}, (Ptr{llama_vocab}, Ptr{Int32}), vocab, n_suppress_tokens)
end

function llama_vocab_fim_pre(vocab)
    ccall((:llama_vocab_fim_pre, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_vocab_fim_suf(vocab)
    ccall((:llama_vocab_fim_suf, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_vocab_fim_mid(vocab)
    ccall((:llama_vocab_fim_mid, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_vocab_fim_pad(vocab)
    ccall((:llama_vocab_fim_pad, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_vocab_fim_rep(vocab)
    ccall((:llama_vocab_fim_rep, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_vocab_fim_sep(vocab)
    ccall((:llama_vocab_fim_sep, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_token_get_text(vocab, token)
    ccall((:llama_token_get_text, libllama), Ptr{Cchar}, (Ptr{llama_vocab}, llama_token), vocab, token)
end

function llama_token_get_score(vocab, token)
    ccall((:llama_token_get_score, libllama), Cfloat, (Ptr{llama_vocab}, llama_token), vocab, token)
end

function llama_token_get_attr(vocab, token)
    ccall((:llama_token_get_attr, libllama), llama_token_attr, (Ptr{llama_vocab}, llama_token), vocab, token)
end

function llama_token_is_eog(vocab, token)
    ccall((:llama_token_is_eog, libllama), Bool, (Ptr{llama_vocab}, llama_token), vocab, token)
end

function llama_token_is_control(vocab, token)
    ccall((:llama_token_is_control, libllama), Bool, (Ptr{llama_vocab}, llama_token), vocab, token)
end

function llama_token_bos(vocab)
    ccall((:llama_token_bos, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_token_eos(vocab)
    ccall((:llama_token_eos, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_token_eot(vocab)
    ccall((:llama_token_eot, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_token_cls(vocab)
    ccall((:llama_token_cls, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_token_sep(vocab)
    ccall((:llama_token_sep, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_token_nl(vocab)
    ccall((:llama_token_nl, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_token_pad(vocab)
    ccall((:llama_token_pad, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_add_bos_token(vocab)
    ccall((:llama_add_bos_token, libllama), Bool, (Ptr{llama_vocab},), vocab)
end

function llama_add_eos_token(vocab)
    ccall((:llama_add_eos_token, libllama), Bool, (Ptr{llama_vocab},), vocab)
end

function llama_token_fim_pre(vocab)
    ccall((:llama_token_fim_pre, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_token_fim_suf(vocab)
    ccall((:llama_token_fim_suf, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_token_fim_mid(vocab)
    ccall((:llama_token_fim_mid, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_token_fim_pad(vocab)
    ccall((:llama_token_fim_pad, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_token_fim_rep(vocab)
    ccall((:llama_token_fim_rep, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_token_fim_sep(vocab)
    ccall((:llama_token_fim_sep, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_vocab_cls(vocab)
    ccall((:llama_vocab_cls, libllama), llama_token, (Ptr{llama_vocab},), vocab)
end

function llama_tokenize(vocab, text, text_len, tokens, n_tokens_max, add_special, parse_special)
    ccall((:llama_tokenize, libllama), Int32, (Ptr{llama_vocab}, Ptr{Cchar}, Int32, Ptr{llama_token}, Int32, Bool, Bool), vocab, text, text_len, tokens, n_tokens_max, add_special, parse_special)
end

function llama_token_to_piece(vocab, token, buf, length, lstrip, special)
    ccall((:llama_token_to_piece, libllama), Int32, (Ptr{llama_vocab}, llama_token, Ptr{Cchar}, Int32, Int32, Bool), vocab, token, buf, length, lstrip, special)
end

function llama_detokenize(vocab, tokens, n_tokens, text, text_len_max, remove_special, unparse_special)
    ccall((:llama_detokenize, libllama), Int32, (Ptr{llama_vocab}, Ptr{llama_token}, Int32, Ptr{Cchar}, Int32, Bool, Bool), vocab, tokens, n_tokens, text, text_len_max, remove_special, unparse_special)
end

function llama_chat_apply_template(tmpl, chat, n_msg, add_ass, buf, length)
    ccall((:llama_chat_apply_template, libllama), Int32, (Ptr{Cchar}, Ptr{llama_chat_message}, Csize_t, Bool, Ptr{Cchar}, Int32), tmpl, chat, n_msg, add_ass, buf, length)
end

function llama_chat_builtin_templates(output, len)
    ccall((:llama_chat_builtin_templates, libllama), Int32, (Ptr{Ptr{Cchar}}, Csize_t), output, len)
end

struct llama_sampler_data
    logits::Ptr{ggml_tensor}
    probs::Ptr{ggml_tensor}
    sampled::Ptr{ggml_tensor}
    candidates::Ptr{ggml_tensor}
end

function llama_set_sampler(ctx, seq_id, smpl)
    ccall((:llama_set_sampler, libllama), Bool, (Ptr{llama_context}, llama_seq_id, Ptr{llama_sampler}), ctx, seq_id, smpl)
end

function llama_sampler_init(iface, ctx)
    ccall((:llama_sampler_init, libllama), Ptr{llama_sampler}, (Ptr{llama_sampler_i}, llama_sampler_context_t), iface, ctx)
end

function llama_sampler_name(smpl)
    ccall((:llama_sampler_name, libllama), Ptr{Cchar}, (Ptr{llama_sampler},), smpl)
end

function llama_sampler_accept(smpl, token)
    ccall((:llama_sampler_accept, libllama), Cvoid, (Ptr{llama_sampler}, llama_token), smpl, token)
end

function llama_sampler_apply(smpl, cur_p)
    ccall((:llama_sampler_apply, libllama), Cvoid, (Ptr{llama_sampler}, Ptr{llama_token_data_array}), smpl, cur_p)
end

function llama_sampler_reset(smpl)
    ccall((:llama_sampler_reset, libllama), Cvoid, (Ptr{llama_sampler},), smpl)
end

function llama_sampler_clone(smpl)
    ccall((:llama_sampler_clone, libllama), Ptr{llama_sampler}, (Ptr{llama_sampler},), smpl)
end

function llama_sampler_copy(src, dst)
    ccall((:llama_sampler_copy, libllama), Cvoid, (Ptr{llama_sampler}, Ptr{llama_sampler}), src, dst)
end

function llama_sampler_free(smpl)
    ccall((:llama_sampler_free, libllama), Cvoid, (Ptr{llama_sampler},), smpl)
end

function llama_sampler_chain_init(params)
    ccall((:llama_sampler_chain_init, libllama), Ptr{llama_sampler}, (llama_sampler_chain_params,), params)
end

function llama_sampler_chain_add(chain, smpl)
    ccall((:llama_sampler_chain_add, libllama), Cvoid, (Ptr{llama_sampler}, Ptr{llama_sampler}), chain, smpl)
end

function llama_sampler_chain_get(chain, i)
    ccall((:llama_sampler_chain_get, libllama), Ptr{llama_sampler}, (Ptr{llama_sampler}, Int32), chain, i)
end

function llama_sampler_chain_n(chain)
    ccall((:llama_sampler_chain_n, libllama), Cint, (Ptr{llama_sampler},), chain)
end

function llama_sampler_chain_remove(chain, i)
    ccall((:llama_sampler_chain_remove, libllama), Ptr{llama_sampler}, (Ptr{llama_sampler}, Int32), chain, i)
end

function llama_sampler_init_greedy()
    ccall((:llama_sampler_init_greedy, libllama), Ptr{llama_sampler}, ())
end

function llama_sampler_init_dist(seed)
    ccall((:llama_sampler_init_dist, libllama), Ptr{llama_sampler}, (UInt32,), seed)
end

function llama_sampler_init_top_k(k)
    ccall((:llama_sampler_init_top_k, libllama), Ptr{llama_sampler}, (Int32,), k)
end

function llama_sampler_init_top_p(p, min_keep)
    ccall((:llama_sampler_init_top_p, libllama), Ptr{llama_sampler}, (Cfloat, Csize_t), p, min_keep)
end

function llama_sampler_init_min_p(p, min_keep)
    ccall((:llama_sampler_init_min_p, libllama), Ptr{llama_sampler}, (Cfloat, Csize_t), p, min_keep)
end

function llama_sampler_init_typical(p, min_keep)
    ccall((:llama_sampler_init_typical, libllama), Ptr{llama_sampler}, (Cfloat, Csize_t), p, min_keep)
end

function llama_sampler_init_temp(t)
    ccall((:llama_sampler_init_temp, libllama), Ptr{llama_sampler}, (Cfloat,), t)
end

function llama_sampler_init_temp_ext(t, delta, exponent)
    ccall((:llama_sampler_init_temp_ext, libllama), Ptr{llama_sampler}, (Cfloat, Cfloat, Cfloat), t, delta, exponent)
end

function llama_sampler_init_xtc(p, t, min_keep, seed)
    ccall((:llama_sampler_init_xtc, libllama), Ptr{llama_sampler}, (Cfloat, Cfloat, Csize_t, UInt32), p, t, min_keep, seed)
end

function llama_sampler_init_top_n_sigma(n)
    ccall((:llama_sampler_init_top_n_sigma, libllama), Ptr{llama_sampler}, (Cfloat,), n)
end

function llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m)
    ccall((:llama_sampler_init_mirostat, libllama), Ptr{llama_sampler}, (Int32, UInt32, Cfloat, Cfloat, Int32), n_vocab, seed, tau, eta, m)
end

function llama_sampler_init_mirostat_v2(seed, tau, eta)
    ccall((:llama_sampler_init_mirostat_v2, libllama), Ptr{llama_sampler}, (UInt32, Cfloat, Cfloat), seed, tau, eta)
end

function llama_sampler_init_grammar(vocab, grammar_str, grammar_root)
    ccall((:llama_sampler_init_grammar, libllama), Ptr{llama_sampler}, (Ptr{llama_vocab}, Ptr{Cchar}, Ptr{Cchar}), vocab, grammar_str, grammar_root)
end

function llama_sampler_init_grammar_lazy(vocab, grammar_str, grammar_root, trigger_words, num_trigger_words, trigger_tokens, num_trigger_tokens)
    ccall((:llama_sampler_init_grammar_lazy, libllama), Ptr{llama_sampler}, (Ptr{llama_vocab}, Ptr{Cchar}, Ptr{Cchar}, Ptr{Ptr{Cchar}}, Csize_t, Ptr{llama_token}, Csize_t), vocab, grammar_str, grammar_root, trigger_words, num_trigger_words, trigger_tokens, num_trigger_tokens)
end

function llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str, grammar_root, trigger_patterns, num_trigger_patterns, trigger_tokens, num_trigger_tokens)
    ccall((:llama_sampler_init_grammar_lazy_patterns, libllama), Ptr{llama_sampler}, (Ptr{llama_vocab}, Ptr{Cchar}, Ptr{Cchar}, Ptr{Ptr{Cchar}}, Csize_t, Ptr{llama_token}, Csize_t), vocab, grammar_str, grammar_root, trigger_patterns, num_trigger_patterns, trigger_tokens, num_trigger_tokens)
end

function llama_sampler_init_penalties(n_vocab, penalty_last_n, penalty_repeat, penalty_freq, penalty_present)
    ccall((:llama_sampler_init_penalties, libllama), Ptr{llama_sampler}, (Int32, Int32, Cfloat, Cfloat, Cfloat), n_vocab, penalty_last_n, penalty_repeat, penalty_freq, penalty_present)
end

function llama_sampler_init_dry(vocab, dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, seq_breakers, num_breakers)
    ccall((:llama_sampler_init_dry, libllama), Ptr{llama_sampler}, (Ptr{llama_vocab}, Cfloat, Cfloat, Int32, Int32, Ptr{Ptr{Cchar}}, Csize_t), vocab, dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, seq_breakers, num_breakers)
end

function llama_sampler_init_adaptive_p(target, decay, seed)
    ccall((:llama_sampler_init_adaptive_p, libllama), Ptr{llama_sampler}, (Cfloat, Cfloat, UInt32), target, decay, seed)
end

function llama_sampler_init_logit_bias(n_vocab, n_logit_bias, logit_bias)
    ccall((:llama_sampler_init_logit_bias, libllama), Ptr{llama_sampler}, (Int32, Int32, Ptr{llama_logit_bias}), n_vocab, n_logit_bias, logit_bias)
end

function llama_sampler_init_infill(vocab)
    ccall((:llama_sampler_init_infill, libllama), Ptr{llama_sampler}, (Ptr{llama_vocab},), vocab)
end

function llama_sampler_get_seed(smpl)
    ccall((:llama_sampler_get_seed, libllama), UInt32, (Ptr{llama_sampler},), smpl)
end

function llama_sampler_sample(smpl, ctx, idx)
    ccall((:llama_sampler_sample, libllama), llama_token, (Ptr{llama_sampler}, Ptr{llama_context}, Int32), smpl, ctx, idx)
end

function llama_split_path(split_path, maxlen, path_prefix, split_no, split_count)
    ccall((:llama_split_path, libllama), Int32, (Ptr{Cchar}, Csize_t, Ptr{Cchar}, Int32, Int32), split_path, maxlen, path_prefix, split_no, split_count)
end

function llama_split_prefix(split_prefix, maxlen, split_path, split_no, split_count)
    ccall((:llama_split_prefix, libllama), Int32, (Ptr{Cchar}, Csize_t, Ptr{Cchar}, Int32, Int32), split_prefix, maxlen, split_path, split_no, split_count)
end

function llama_print_system_info()
    ccall((:llama_print_system_info, libllama), Ptr{Cchar}, ())
end

function llama_log_get(log_callback, user_data)
    ccall((:llama_log_get, libllama), Cvoid, (Ptr{ggml_log_callback}, Ptr{Ptr{Cvoid}}), log_callback, user_data)
end

function llama_log_set(log_callback, user_data)
    ccall((:llama_log_set, libllama), Cvoid, (ggml_log_callback, Ptr{Cvoid}), log_callback, user_data)
end

struct llama_perf_context_data
    t_start_ms::Cdouble
    t_load_ms::Cdouble
    t_p_eval_ms::Cdouble
    t_eval_ms::Cdouble
    n_p_eval::Int32
    n_eval::Int32
    n_reused::Int32
end

struct llama_perf_sampler_data
    t_sample_ms::Cdouble
    n_sample::Int32
end

function llama_perf_context(ctx)
    ccall((:llama_perf_context, libllama), llama_perf_context_data, (Ptr{llama_context},), ctx)
end

function llama_perf_context_print(ctx)
    ccall((:llama_perf_context_print, libllama), Cvoid, (Ptr{llama_context},), ctx)
end

function llama_perf_context_reset(ctx)
    ccall((:llama_perf_context_reset, libllama), Cvoid, (Ptr{llama_context},), ctx)
end

function llama_perf_sampler(chain)
    ccall((:llama_perf_sampler, libllama), llama_perf_sampler_data, (Ptr{llama_sampler},), chain)
end

function llama_perf_sampler_print(chain)
    ccall((:llama_perf_sampler_print, libllama), Cvoid, (Ptr{llama_sampler},), chain)
end

function llama_perf_sampler_reset(chain)
    ccall((:llama_perf_sampler_reset, libllama), Cvoid, (Ptr{llama_sampler},), chain)
end

# typedef bool ( * llama_opt_param_filter ) ( const struct ggml_tensor * tensor , void * userdata )
const llama_opt_param_filter = Ptr{Cvoid}

function llama_opt_param_filter_all(tensor, userdata)
    ccall((:llama_opt_param_filter_all, libllama), Bool, (Ptr{ggml_tensor}, Ptr{Cvoid}), tensor, userdata)
end

struct llama_opt_params
    n_ctx_train::UInt32
    param_filter::llama_opt_param_filter
    param_filter_ud::Ptr{Cvoid}
    get_opt_pars::ggml_opt_get_optimizer_params
    get_opt_pars_ud::Ptr{Cvoid}
    optimizer_type::ggml_opt_optimizer_type
end

function llama_opt_init(lctx, model, lopt_params)
    ccall((:llama_opt_init, libllama), Cvoid, (Ptr{llama_context}, Ptr{llama_model}, llama_opt_params), lctx, model, lopt_params)
end

function llama_opt_epoch(lctx, dataset, result_train, result_eval, idata_split, callback_train, callback_eval)
    ccall((:llama_opt_epoch, libllama), Cvoid, (Ptr{llama_context}, ggml_opt_dataset_t, ggml_opt_result_t, ggml_opt_result_t, Int64, ggml_opt_epoch_callback, ggml_opt_epoch_callback), lctx, dataset, result_train, result_eval, idata_split, callback_train, callback_eval)
end

# Skipping MacroDefinition: GGML_API extern

const GGML_FILE_MAGIC = 0x67676d6c

const GGML_FILE_VERSION = 2

const GGML_QNT_VERSION = 2

const GGML_QNT_VERSION_FACTOR = 1000

const GGML_MAX_DIMS = 4

const GGML_MAX_PARAMS = 2048

const GGML_MAX_SRC = 10

const GGML_MAX_N_THREADS = 512

const GGML_MAX_OP_PARAMS = 64

const GGML_MAX_NAME = 64

const GGML_DEFAULT_N_THREADS = 4

const GGML_DEFAULT_GRAPH_SIZE = 2048

const GGML_MEM_ALIGN = 16

const GGML_EXIT_SUCCESS = 0

const GGML_EXIT_ABORTED = 1

const GGML_ROPE_TYPE_NORMAL = 0

const GGML_ROPE_TYPE_NEOX = 2

const GGML_ROPE_TYPE_MROPE = 8

const GGML_ROPE_TYPE_VISION = 24

const GGML_ROPE_TYPE_IMROPE = 40

const GGML_MROPE_SECTIONS = 4

const GGML_N_TASKS_MAX = -1

# Skipping MacroDefinition: GGML_BACKEND_API extern

const GGML_BACKEND_META_MAX_DEVICES = 16

const GGUF_MAGIC = "GGUF"

const GGUF_VERSION = 3

const GGUF_KEY_GENERAL_ALIGNMENT = "general.alignment"

const GGUF_DEFAULT_ALIGNMENT = 32

const LLAMA_DEFAULT_SEED = 0xffffffff

const LLAMA_TOKEN_NULL = -1

const LLAMA_FILE_MAGIC_GGLA = Cuint(0x67676c61)

const LLAMA_FILE_MAGIC_GGSN = Cuint(0x6767736e)

const LLAMA_FILE_MAGIC_GGSQ = Cuint(0x67677371)

const LLAMA_SESSION_MAGIC = LLAMA_FILE_MAGIC_GGSN

const LLAMA_SESSION_VERSION = 9

const LLAMA_STATE_SEQ_MAGIC = LLAMA_FILE_MAGIC_GGSQ

const LLAMA_STATE_SEQ_VERSION = 2

const LLAMA_STATE_SEQ_FLAGS_NONE = 0

const LLAMA_STATE_SEQ_FLAGS_SWA_ONLY = 1

const LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY = 1

const LLAMA_STATE_SEQ_FLAGS_ON_DEVICE = 2

end # module
