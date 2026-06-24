// windows_lite_test_stubs.cpp
//
// Minimal stubs for ROCr core symbols the WindowsLiteDriver translation unit
// REFERENCES (so it links) but the lite:: dispatch self-test never EXERCISES.
// Populated empirically from the linker's unresolved-external list. Keeping
// these here lets the driver-kernel test link without dragging the full ROCr
// runtime (agents, memory pools, Runtime singleton) into the test binary.
