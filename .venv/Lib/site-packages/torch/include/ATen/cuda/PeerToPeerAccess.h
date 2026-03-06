#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
#include <c10/macros/Macros.h>
#include <c10/core/Device.h>
#include <cstdint>

namespace at::cuda {
namespace detail {
void init_p2p_access_cache(int64_t num_devices);
}

TORCH_CUDA_CPP_API bool get_p2p_access(c10::DeviceIndex source_dev, c10::DeviceIndex dest_dev);
TORCH_CUDA_CPP_API bool get_fabric_access(c10::DeviceIndex device);

}  // namespace at::cuda

#else
#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
