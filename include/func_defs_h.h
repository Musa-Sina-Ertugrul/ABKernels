#ifndef __FUNC_DEF_H__
#define __FUNC_DEF_H__

#define CREATE_CALL_FUNCTION_X2_2INPUT_H(func_name) \
torch::Tensor \
func_name(torch::Tensor a,torch::Tensor b);

#define CREATE_CALL_FUNCTION_X2_H(func_name) \
torch::Tensor \
func_name(torch::Tensor input);

#define CREATE_ACTIVATION_FUNC_INNER_H(func_name,dtype) \
__device__ \
__forceinline__ \
dtype \
func_name(dtype input);

#define CREATE_FUNC_INTERMEADIATE_2INPUT_H(func_name,dtype) \
__global__ \
void \
func_name(dtype* array_left, dtype* array_right,dtype* output,uint64_t len);

#define CREATE_FUNC_INTERMEADIATE_H(func_name,dtype) \
__global__ \
void \
func_name(dtype* input,uint64_t len);

#endif