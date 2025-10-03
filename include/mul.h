#ifndef __MUL__
#define __MUL__

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/all.h>
#include "func_defs_h.h"

CREATE_FUNC_INTERMEADIATE_2INPUT_H(mul_bfx2,nv_bfloat162)
CREATE_FUNC_INTERMEADIATE_2INPUT_H(mul_fx2,half2)

CREATE_CALL_FUNCTION_X2_2INPUT_H(mul)

#endif