// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

// Each compilation exports one module symbol. Two independent payloads with the
// same target exercise pack selection by both target and logical module name.
#if THEROCK_MODULE_RELU
#define THEROCK_MODULE_SYMBOL therock_module_relu
#else
#define THEROCK_MODULE_SYMBOL therock_module_saxpy
#endif

extern "C" __global__ void THEROCK_MODULE_SYMBOL(const float *x, const float *y,
                                                 float *output, float alpha,
                                                 unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    float value = alpha * x[i] + y[i];
#if THEROCK_MODULE_RELU
    value -= 0.125f;
    value = value > 0.0f ? value : 0.0f;
#endif
    output[i] = value;
  }
}
