// windows_lite_driver_kernel_test.cpp
//
// Dispatch a real compiled kernel (fill_kernel_raw.co) through the REAL ROCr
// WindowsLiteDriver object -- the core::Driver + lite::DirectQueuePlatform that
// ships in libhsa-runtime64.dll -- rather than the standalone test platform.
// Proves the committed driver's lite:: dispatch path end-to-end on HW.
#include <cstdio>
#include <string>

#include "core/inc/amd_windows_lite_driver.h"

int main() {
  // Heap-allocate + intentionally leak: WindowsLiteDriver is a PIMPL
  // (unique_ptr<WddmLiteState>, an incomplete type in the header) with an
  // implicit dtor, so destroying it from this TU would need the complete type.
  // A one-shot test process leaks harmlessly at exit.
  auto* driver = new rocr::AMD::WindowsLiteDriver(std::string("amdgpu_mcdm"));
  std::printf("windows_lite_driver_kernel_test: dispatching fill_kernel via the "
              "real WindowsLiteDriver\n");
  hsa_status_t r = driver->DispatchKernelSelfTest();
  std::printf("windows_lite_driver_kernel_test: DispatchKernelSelfTest -> %u\n", r);
  return r == HSA_STATUS_SUCCESS ? 0 : 1;
}
