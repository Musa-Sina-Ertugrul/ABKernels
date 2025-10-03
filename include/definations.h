#ifndef __DEF__
#define __DEF__

#ifndef THREAD_COUNT
    #define THREAD_COUNT 256
#endif

#define BF_ptr(value) reinterpret_cast<nv_bfloat16*>(value)
#define BFX2_ptr(value) reinterpret_cast<nv_bfloat162*>(value)
#define F_ptr(value) reinterpret_cast<half*>(value)
#define FX2_ptr(value) reinterpret_cast<half2*>(value)

#define STRINGIFY(x) #x
#define CONCAT(a, b) a##b

#endif
