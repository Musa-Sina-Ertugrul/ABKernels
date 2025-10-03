#ifndef __GELU__
#define __GELU__

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/all.h>
#include "func_defs_h.h"

CREATE_ACTIVATION_FUNC_INNER(__gelu_bfx2,,__hmul2(
        __hmul2(__float2bfloat162_rn(0.5f),input),
        __hadd2(__float2bfloat162_rn(1.0f),
            h2tanh(
                __hmul2(__float2bfloat162_rn(SQRT_2_OVER_PI),
                    __hadd2(input,
                        __hmul2(__float2bfloat162_rn(0.044715f),
                            __hmul2(__hmul2(input, input), input))))))),nv_bfloat162)

CREATE_ACTIVATION_FUNC_INNER(__gelu_fx2,, __hmul2(
        __hmul2(__float2half2_rn(0.5f),input),
        __hadd2(__float2half2_rn(1.0f),
            h2tanh(
                __hmul2(__float2half2_rn(SQRT_2_OVER_PI),
                    __hadd2(input,
                        __hmul2(__float2half2_rn(0.044715f),
                            __hmul2(__hmul2(input, input), input))))))),half2)

// Formula: grad_out * 0.5 * [(1 + tanh(z)) + x * c1 * sech²(z) * (1 + 3*c2*x²)]
// where z = c1*(x + c2*x³), c1=0.45, c2=0.044715
CREATE_ACTIVATION_FUNC_INNER(__gelu_backward_bfx2,
    nv_bfloat162 input_tanh = h2tanh(input);
    nv_bfloat162 input_sech = h2sqrt(__hadd2(__float2bfloat162_rn(1.0f),input_tanh));,
    (__hmul2(__float2bfloat162_rn(0.5f),
        __hadd2(
            __hadd2(__float2bfloat162_rn(0.5f),input_tanh),
                __hmul2(input,
                    __hmul2(__float2bfloat162_rn(SQRT_2_OVER_PI),
                        __hmul2(
                            __hmul2(input_sech,input_sech),
                                __hadd2(__float2bfloat162_rn(1.0f),
                                    __hmul2(__float2bfloat162_rn(3.0f),
                                            __hmul2(__float2bfloat162_rn(0.044715f),
                                                    __hmul2(input,input)))))))))), nv_bfloat162)


// Formula: grad_out * 0.5 * [(1 + tanh(z)) + x * c1 * sech²(z) * (1 + 3*c2*x²)]
CREATE_ACTIVATION_FUNC_INNER(__gelu_backward_fx2,
    half2 input_tanh = h2tanh(input);
    half2 input_sech = h2sqrt(__hadd2(__float2half2_rn(1.0f),input_tanh));,
    (__hmul2(__float2half2_rn(0.5f),
        __hadd2(
            __hadd2(__float2half2_rn(0.5f),input_tanh),
                __hmul2(input,
                    __hmul2(__float2half2_rn(SQRT_2_OVER_PI),
                        __hmul2(
                            __hmul2(input_sech,input_sech),
                                __hadd2(__float2half2_rn(1.0f),
                                    __hmul2(__float2half2_rn(3.0f),
                                            __hmul2(__float2half2_rn(0.044715f),
                                                    __hmul2(input,input)))))))))), half2)

CREATE_FUNC_INTERMEADIATE_H(gelu_bfx2,nv_bfloat162)
CREATE_FUNC_INTERMEADIATE_H(gelu_fx2,half2)
CREATE_FUNC_INTERMEADIATE_H(gelu_backward_bfx2,nv_bfloat162)
CREATE_FUNC_INTERMEADIATE_H(gelu_backward_fx2,half2)
CREATE_CALL_FUNCTION_X2_H(gelu)
CREATE_CALL_FUNCTION_X2_H(gelu_backward)

#endif