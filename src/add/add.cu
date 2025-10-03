#ifndef __ADD__
#define __ADD__

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include "../utils/func_defs.cu"
#include "../utils/definations.cu"

CREATE_FUNC_INTERMEADIATE_2INPUT(add_bfx2,
        nv_bfloat162* local_left = BFX2_ptr(array_left);
        nv_bfloat162* local_right = BFX2_ptr(array_right);
        BFX2_ptr(output)[index] = __hadd2(local_left[index],local_right[index]);,nv_bfloat16)

CREATE_FUNC_INTERMEADIATE_2INPUT(add_fx2,
        half2* local_left = FX2_ptr(array_left);
        half2* local_right = FX2_ptr(array_right);
        FX2_ptr(output)[index] = __hadd2(local_left[index],local_right[index]);, half)

CREATE_CALL_FUNCTION_X2_2INPUT(add)

#endif
