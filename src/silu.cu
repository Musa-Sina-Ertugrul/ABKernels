#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "func_defs.h"
#include "definations.h"
#include "sigmoid.h"
#include "silu.h"

CREATE_FUNC_INTERMEADIATE(silu_bfx2,        
        nv_bfloat162* local_input = BFX2_ptr(input);
        local_input[index] = __silu_bfx2(local_input[index]);,nv_bfloat16)

CREATE_FUNC_INTERMEADIATE(silu_fx2,
        half2* local_input = FX2_ptr(input);
        local_input[index] = __silu_fx2(local_input[index]);,half)

CREATE_FUNC_INTERMEADIATE(silu_backward_bfx2,        
        nv_bfloat162* local_input = BFX2_ptr(input);
        local_input[index] = __silu_backward_bfx2(local_input[index]);,nv_bfloat16)

CREATE_FUNC_INTERMEADIATE(silu_backward_fx2,
        half2* local_input = FX2_ptr(input);
        local_input[index] = __silu_backward_fx2(local_input[index]);,half)

CREATE_CALL_FUNCTION_X2(silu)

CREATE_CALL_FUNCTION_X2(silu_backward)