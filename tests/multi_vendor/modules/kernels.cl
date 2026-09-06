// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Portable OpenCL C supplies the native Level Zero SPIR-V validation payload.
// This tests the module ABI; HIP source portability uses the separate chipStar
// consumer. Build each module separately to exercise disjoint pack lookup.
#ifndef THEROCK_MODULE_RELU
#error "Set THEROCK_MODULE_RELU to select the module"
#endif

#if THEROCK_MODULE_RELU
#define THEROCK_MODULE_ENTRY therock_module_relu
#else
#define THEROCK_MODULE_ENTRY therock_module_saxpy
#endif

__kernel void THEROCK_MODULE_ENTRY(__global const float *x,
                                   __global const float *y,
                                   __global float *output, float alpha,
                                   unsigned int count) {
  const size_t index = get_global_id(0);
  if (index < count) {
    float value = alpha * x[index] + y[index];
#if THEROCK_MODULE_RELU
    value = fmax(0.0f, value - 0.125f);
#endif
    output[index] = value;
  }
}
