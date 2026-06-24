// hsa_discovery_test.cpp -- stock HSA API: does hsa_iterate_agents find the
// gfx1201 GPU agent (WindowsGpuAgent) through the real libhsa-runtime64.dll?
#include <cstdio>
#include <cstring>
#include "hsa.h"

static hsa_status_t agent_cb(hsa_agent_t agent, void* data) {
  char name[64] = {0};
  hsa_device_type_t dt = (hsa_device_type_t)-1;
  hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
  hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dt);
  std::printf("agent handle=0x%llx name='%s' device_type=%d\n",
              (unsigned long long)agent.handle, name, (int)dt);
  if (dt == HSA_DEVICE_TYPE_GPU) {
    (*static_cast<int*>(data))++;
    hsa_isa_t isa{};
    if (hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &isa) == HSA_STATUS_SUCCESS) {
      char isaname[80] = {0};
      hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, isaname);
      std::printf("  GPU ISA: %s\n", isaname);
    }
  }
  return HSA_STATUS_SUCCESS;
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered so output survives a crash
  hsa_status_t s = hsa_init();
  std::printf("hsa_init -> %d\n", s);
  if (s != HSA_STATUS_SUCCESS) return 2;
  int gpus = 0;
  hsa_iterate_agents(agent_cb, &gpus);
  std::printf("GPU agents found: %d\n", gpus);
  hsa_shut_down();
  std::printf("HSA_DISCOVERY %s\n", gpus > 0 ? "PASS" : "FAIL (no GPU agent)");
  return gpus > 0 ? 0 : 1;
}
