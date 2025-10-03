#ifndef __FUNC_DEF__
#define __FUNC_DEF__

#include "./definations.cu"

#define CREATE_CALL_FUNCTION_X2_2INPUT(func_name) \
torch::Tensor \
func_name(torch::Tensor a,torch::Tensor b){\
    TORCH_CHECK(a.numel() == b.numel(),"Input arrays have different length");\
    TORCH_CHECK(a.device().index()==b.device().index(),"Input arrays have different devices");\
    TORCH_CHECK(a.dtype()==b.dtype(),"Input arrays have different dtypes");\
    TORCH_CHECK(a.numel() % 2 == 0, std::string("There is no such option for ") + STRINGIFY(func_name) + " two array use torch." + STRINGIFY(func_name));\
    torch::Tensor output = torch::zeros_like(a);\
    int device = a.device().index();\
    int block_count = (a.numel() / (2*THREAD_COUNT)) + 1;\
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();\
    switch (a.scalar_type())\
    {\
    case torch::kBFloat16:\
        CONCAT(func_name,_bfx2)<<<block_count,THREAD_COUNT, 0, stream>>>(BF_ptr(a.data_ptr()),BF_ptr(b.data_ptr()),BF_ptr(output.data_ptr()),a.numel()/2);\
        break;\
    case torch::kFloat16:\
        CONCAT(func_name,_fx2)<<<block_count,THREAD_COUNT, 0, stream>>>(F_ptr(a.data_ptr()),F_ptr(b.data_ptr()),F_ptr(output.data_ptr()),a.numel()/2);\
        break;\
    default:\
        TORCH_CHECK(false, std::string("There is no such option for ") + STRINGIFY(func_name) + " two array use torch." + STRINGIFY(func_name));\
        break;\
    }\
    cudaStreamSynchronize(stream);\
    return output;\
}


#define CREATE_CALL_FUNCTION_X2(func_name) \
torch::Tensor \
func_name(torch::Tensor input){\
    TORCH_CHECK(input.numel() % 2 == 0, std::string("There is no such option for ") + STRINGIFY(gelu) + " two array use torch." + STRINGIFY(gelu));\
    int device = input.device().index();\
    int block_count = (input.numel() / (2*THREAD_COUNT)) + 1;\
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();\
    switch (input.scalar_type())\
    {\
    case torch::kBFloat16:\
        CONCAT(func_name,_bfx2)<<<block_count,THREAD_COUNT, 0, stream>>>(BF_ptr(input.data_ptr()),input.numel()/2);\
        break;\
    case torch::kFloat16:\
        CONCAT(func_name,_fx2)<<<block_count,THREAD_COUNT, 0, stream>>>(F_ptr(input.data_ptr()),input.numel()/2);\
        break;\
    default:\
        TORCH_CHECK(false, std::string("There is no such option for ") + STRINGIFY(func_name) + " two array use torch." + STRINGIFY(func_name));\
        break;\
    }\
    cudaStreamSynchronize(stream);\
    return input;\
}

#define CREATE_ACTIVATION_FUNC_INNER(func_name,pre_calculation,calculation,dtype) \
static \
__device__ \
__forceinline__ \
dtype \
func_name(dtype input) { \
    do\
    {\
        pre_calculation\
        return calculation;\
    } while (0);\
}

#define CREATE_FUNC_INTERMEADIATE_2INPUT(func_name,inside,dtype) \
__global__ \
void \
func_name(dtype* array_left, dtype* array_right,dtype* output,uint64_t len){ \
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index < len){ \
        inside \
    }\
}

#define CREATE_FUNC_INTERMEADIATE(func_name,inside,dtype) \
__global__ \
void \
func_name(dtype* input,u_int64_t len){ \
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index< len){ \
        inside \
    } \
}


#endif