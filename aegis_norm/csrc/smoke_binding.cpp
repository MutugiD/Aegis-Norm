#include <torch/extension.h>

torch::Tensor smoke_cuda(const torch::Tensor& input);

torch::Tensor smoke(const torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "smoke requires a CUDA tensor");
  TORCH_CHECK(input.layout() == c10::kStrided && input.is_contiguous(),
              "smoke requires contiguous strided input");
  TORCH_CHECK(input.scalar_type() == at::kFloat, "smoke requires FP32 input");
  TORCH_CHECK(!at::GradMode::is_enabled() || !input.requires_grad(),
              "smoke is inference-only");
  return smoke_cuda(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &smoke, "Native build smoke: y = 3 * x (not RMSNorm)");
}
