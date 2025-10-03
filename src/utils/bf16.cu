#ifndef __BF16__
#define __BF16__

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "./func_defs.cu"

CREATE_ACTIVATION_FUNC_INNER(bf16_tanh,
    float2 input_fp32 = __bfloat1622float2(input);
    input_fp32.x = tanhf(input_fp32.x);
    input_fp32.y = tanhf(input_fp32.y);
    ,__float22bfloat162_rn(input_fp32),nv_bfloat162
)

#endif