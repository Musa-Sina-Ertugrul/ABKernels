#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "func_defs.h"
#include "definations.h"
#include "sigmoid.h"

CREATE_FUNC_INTERMEADIATE(sigmoid_bfx2,        
        nv_bfloat162* local_input = BFX2_ptr(input);
        local_input[index] = __sigmoid_bfx2(local_input[index]);,nv_bfloat16)

CREATE_FUNC_INTERMEADIATE(sigmoid_fx2,
        half2* local_input = FX2_ptr(input);
        local_input[index] = __sigmoid_fx2(local_input[index]);,half)

CREATE_FUNC_INTERMEADIATE(sigmoid_backward_bfx2,        
        nv_bfloat162* local_input = BFX2_ptr(input);
        local_input[index] = __sigmoid_backward_bfx2(local_input[index]);,nv_bfloat16)

CREATE_FUNC_INTERMEADIATE(sigmoid_backward_fx2,
        half2* local_input = FX2_ptr(input);
        local_input[index] = __sigmoid_backward_fx2(local_input[index]);,half)

CREATE_CALL_FUNCTION_X2(sigmoid)

CREATE_CALL_FUNCTION_X2(sigmoid_backward)