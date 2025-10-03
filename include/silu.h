#ifndef __SILU__
#define __SILU__

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/all.h>
#include "func_defs_h.h"

CREATE_ACTIVATION_FUNC_INNER(__silu_bfx2,,
    __hmul2(input,__sigmoid_bfx2(input)),nv_bfloat162)

CREATE_ACTIVATION_FUNC_INNER(__silu_fx2,,
    __hmul2(input,__sigmoid_fx2(input)),half2)

CREATE_ACTIVATION_FUNC_INNER(__silu_backward_bfx2,
    nv_bfloat162 input_silu = __silu_bfx2(input);
    nv_bfloat162 input_sigmoid = __sigmoid_bfx2(input);,
    __hadd2(input_sigmoid,__hmul2(input_silu,__hsub2(__float2bfloat162_rn(1.0f),input_sigmoid))),nv_bfloat162)

CREATE_ACTIVATION_FUNC_INNER(__silu_backward_fx2,
    half2 input_silu = __silu_fx2(input);
    half2 input_sigmoid = __sigmoid_fx2(input);,
    __hadd2(input_sigmoid,__hmul2(input_silu,__hsub2(__float2half2_rn(1.0f),input_sigmoid))),half2)

CREATE_FUNC_INTERMEADIATE_H(silu_bfx2,nv_bfloat162)
CREATE_FUNC_INTERMEADIATE_H(silu_fx2,half2)
CREATE_FUNC_INTERMEADIATE_H(silu_backward_bfx2,nv_bfloat162)
CREATE_FUNC_INTERMEADIATE_H(silu_backward_fx2,half2)
CREATE_CALL_FUNCTION_X2_H(silu)
CREATE_CALL_FUNCTION_X2_H(silu_backward)

#endif