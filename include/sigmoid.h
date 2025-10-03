#ifndef __SIGMOID__
#define __SIGMOID__

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/all.h>
#include "func_defs_h.h"

CREATE_ACTIVATION_FUNC_INNER(__sigmoid_bfx2,,
    __h2div(__float2bfloat162_rn(1.0f),
    __hadd2(__float2bfloat162_rn(1.0f),
    h2exp(__hneg2(input)))),nv_bfloat162)

CREATE_ACTIVATION_FUNC_INNER(__sigmoid_fx2,,
    __h2div(__float2half2_rn(1.0f),
    __hadd2(__float2half2_rn(1.0f),
    h2exp(__hneg2(input)))),half2)

CREATE_ACTIVATION_FUNC_INNER(__sigmoid_backward_bfx2,
    nv_bfloat162 input_sigmoid = __sigmoid_bfx2(input);,
    __hmul2(input_sigmoid,__hsub2(__float2bfloat162_rn(1.0f),input_sigmoid)),nv_bfloat162)

CREATE_ACTIVATION_FUNC_INNER(__sigmoid_backward_fx2,
    half2 input_sigmoid = __sigmoid_fx2(input);,
    __hmul2(input_sigmoid,__hsub2(__float2half2_rn(1.0f),input_sigmoid)),half2)

CREATE_FUNC_INTERMEADIATE_H(sigmoid_bfx2,nv_bfloat162)
CREATE_FUNC_INTERMEADIATE_H(sigmoid_fx2,half2)
CREATE_FUNC_INTERMEADIATE_H(sigmoid_backward_bfx2,nv_bfloat162)
CREATE_FUNC_INTERMEADIATE_H(sigmoid_backward_fx2,half2)
CREATE_CALL_FUNCTION_X2_H(sigmoid)
CREATE_CALL_FUNCTION_X2_H(sigmoid_backward)

#endif