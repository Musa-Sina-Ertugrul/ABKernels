#ifndef __GELU__
#define __GELU__

#include <math.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "../utils/func_defs.cu"
#include "../utils/definations.cu"
#include "../utils/bf16.cu"

CREATE_ACTIVATION_FUNC_INNER(__gelu_bfx2,,__hmul2(
        __hmul2(__float2bfloat162_rn(0.5f),input),
        __hadd2(__float2bfloat162_rn(1.0f),
            h2tanh(
                __hmul2(__float2bfloat162_rn(0.45f),
                    __hadd2(input,
                        __hmul2(__float2bfloat162_rn(0.044715f),
                            __hmul2(__hmul2(input, input), input))))))),nv_bfloat162)

CREATE_ACTIVATION_FUNC_INNER(__gelu_fx2,, __hmul2(
        __hmul2(__float2half2_rn(0.5f),input),
        __hadd2(__float2half2_rn(1.0f),
            h2tanh(
                __hmul2(__float2half2_rn(0.45f),
                    __hadd2(input,
                        __hmul2(__float2half2_rn(0.044715f),
                            __hmul2(__hmul2(input, input), input))))))),half2)

// GELU backward for bfloat162
// Formula: grad_out * 0.5 * [(1 + tanh(z)) + x * c1 * sech²(z) * (1 + 3*c2*x²)]
// where z = c1*(x + c2*x³), c1=0.45, c2=0.044715
CREATE_ACTIVATION_FUNC_INNER(__gelu_backward_bfx2,// Compute x² and x³
    nv_bfloat162 x2 = __hmul2(input, input);
    nv_bfloat162 x3 = __hmul2(x2, input);

    // Compute z = c1*(x + c2*x³)
    nv_bfloat162 inner = __hadd2(input,
                                  __hmul2(__float2bfloat162_rn(0.044715f), x3));
    nv_bfloat162 z = __hmul2(__float2bfloat162_rn(0.45f), inner);

    // Compute tanh(z)
    nv_bfloat162 tanh_z = bf16_tanh(z);

    // Compute sech²(z) = 1 - tanh²(z)
    nv_bfloat162 tanh2_z = __hmul2(tanh_z, tanh_z);
    nv_bfloat162 sech2_z = __hsub2(__float2bfloat162_rn(1.0f), tanh2_z);

    // Compute (1 + 3*c2*x²)
    nv_bfloat162 term_x2 = __hadd2(__float2bfloat162_rn(1.0f),
                                    __hmul2(__float2bfloat162_rn(3.0f * 0.044715f), x2));

    // Compute x * c1 * sech²(z) * (1 + 3*c2*x²)
    nv_bfloat162 term2 = __hmul2(__hmul2(__hmul2(input, __float2bfloat162_rn(0.45f)),
                                          sech2_z),
                                  term_x2);

    // Compute (1 + tanh(z)) + term2
    nv_bfloat162 sum_terms = __hadd2(__hadd2(__float2bfloat162_rn(1.0f), tanh_z),
                                      term2);

    // Multiply by 0.5
    nv_bfloat162 grad_gelu = __hmul2(__float2bfloat162_rn(0.5f), sum_terms);,grad_gelu, nv_bfloat162)


// GELU backward for half2 (can use built-in h2tanh!)
// Formula: grad_out * 0.5 * [(1 + tanh(z)) + x * c1 * sech²(z) * (1 + 3*c2*x²)]
CREATE_ACTIVATION_FUNC_INNER(__gelu_backward_fx2,// Compute x² and x³
    half2 x2 = __hmul2(input, input);
    half2 x3 = __hmul2(x2, input);

    // Compute z = c1*(x + c2*x³)
    half2 inner = __hadd2(input,
                           __hmul2(__float2half2_rn(0.044715f), x3));
    half2 z = __hmul2(__float2half2_rn(0.45f), inner);

    // Compute tanh(z) - half2 HAS built-in tanh!
    half2 tanh_z = h2tanh(z);

    // Compute sech²(z) = 1 - tanh²(z)
    half2 tanh2_z = __hmul2(tanh_z, tanh_z);
    half2 sech2_z = __hsub2(__float2half2_rn(1.0f), tanh2_z);

    // Compute (1 + 3*c2*x²)
    half2 term_x2 = __hadd2(__float2half2_rn(1.0f),
                             __hmul2(__float2half2_rn(3.0f * 0.044715f), x2));

    // Compute x * c1 * sech²(z) * (1 + 3*c2*x²)
    half2 term2 = __hmul2(__hmul2(__hmul2(input, __float2half2_rn(0.45f)),
                                   sech2_z),
                           term_x2);

    // Compute (1 + tanh(z)) + term2
    half2 sum_terms = __hadd2(__hadd2(__float2half2_rn(1.0f), tanh_z),
                               term2);

    // Multiply by 0.5
    half2 grad_gelu = __hmul2(__float2half2_rn(0.5f), sum_terms);,grad_gelu,half2)

CREATE_FUNC_INTERMEADIATE(gelu_bfx2,        
        nv_bfloat162* local_input = BFX2_ptr(input);
        local_input[index] = __gelu_bfx2(local_input[index]);,nv_bfloat16)

CREATE_FUNC_INTERMEADIATE(gelu_fx2,
        half2* local_input = FX2_ptr(input);
        local_input[index] = __gelu_fx2(local_input[index]);,half)

CREATE_FUNC_INTERMEADIATE(gelu_backward_bfx2,        
        nv_bfloat162* local_input = BFX2_ptr(input);
        local_input[index] = __gelu_backward_bfx2(local_input[index]);,nv_bfloat16)

CREATE_FUNC_INTERMEADIATE(gelu_backward_fx2,
        half2* local_input = FX2_ptr(input);
        local_input[index] = __gelu_backward_fx2(local_input[index]);,half)

CREATE_CALL_FUNCTION_X2(gelu)

CREATE_CALL_FUNCTION_X2(gelu_backward)

#endif