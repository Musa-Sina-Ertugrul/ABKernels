#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "func_defs.h"
#include "definations.h"
#include "constants.h"
#include "gelu.h"

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