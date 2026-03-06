#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>

namespace at::xpu {
namespace detail {
void init_p2p_access_cache(c10::DeviceIndex num_devices);
} // namespace detail

TORCH_XPU_API bool get_p2p_access(
    c10::DeviceIndex dev,
    c10::DeviceIndex dev_to_access);

} // namespace at::xpu

#else
#error "This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."
#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
