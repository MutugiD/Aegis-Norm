#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include <cmath>
#include <limits>

at::Tensor rmsnorm_launch(const at::Tensor& x, const at::Tensor& weight, float eps);

at::Tensor rmsnorm_checked(const at::Tensor& x, const at::Tensor& weight, double eps) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "x and weight must be CUDA tensors");
  TORCH_CHECK(x.device() == weight.device(), "x and weight must be on the same device");
  TORCH_CHECK(x.layout() == c10::kStrided && weight.layout() == c10::kStrided,
              "x and weight must use strided layout");
  TORCH_CHECK(x.dim() >= 1 && x.size(-1) > 0 && x.size(-1) <= 65536,
              "x must have rank >= 1 and width 1..65536");
  const int64_t width = x.size(-1);
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == width, "weight must have shape [H]");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "native RMSNorm requires contiguous inputs");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf,
              "native RMSNorm supports FP16 and FP32");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "x and weight must have the same dtype");
  TORCH_CHECK(std::isfinite(eps) && eps > 0 &&
              eps <= static_cast<double>(std::numeric_limits<float>::max()),
              "eps must be positive finite FP32");
  const float eps32 = static_cast<float>(eps);
  TORCH_CHECK(eps32 > 0, "eps must be representable as positive FP32");
  TORCH_CHECK(!at::GradMode::is_enabled() || (!x.requires_grad() && !weight.requires_grad()),
              "native RMSNorm is inference-only; use reference for active gradients");
  TORCH_CHECK(x.numel() / width <= std::numeric_limits<int32_t>::max(),
              "row count exceeds the supported CUDA grid limit");
  const auto* properties = at::cuda::getDeviceProperties(x.get_device());
  TORCH_CHECK(properties->major == 7 && properties->minor == 5,
              "native RMSNorm currently targets compute capability 7.5");
  return rmsnorm_launch(x, weight, eps32);
}

TORCH_LIBRARY(aegis_norm, library) {
  library.def("rms_norm(Tensor x, Tensor weight, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(aegis_norm, CUDA, library) {
  library.impl("rms_norm", &rmsnorm_checked);
}

// Reject active gradients explicitly rather than creating a silent wrong backward.
TORCH_LIBRARY_IMPL(aegis_norm, Autograd, library) {
  library.impl("rms_norm", &rmsnorm_checked);
}
