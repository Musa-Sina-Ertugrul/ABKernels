from torch.utils.cpp_extension import load

#NOTE: export CUDA_HOME=/usr/local/cuda

_kernel = load("_kernel",
        sources=["src/module.cu",
                "src/utils/constants.cu",
                "src/utils/definations.cu",
                "src/utils/func_defs.cu",
                "src/add/add.cu",
                "src/mul/mul.cu",
                "src/gelu/gelu.cu"
                ],
        extra_cuda_cflags=["-O3",
                                "--use_fast_math",
                                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                                "-U__CUDA_NO_HALF_OPERATORS__",
                                "-U__CUDA_NO_HALF_CONVERSIONS__"],
        verbose=True)