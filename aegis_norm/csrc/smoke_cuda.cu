#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <cstdint>

__global__ void triple_kernel(const float* input, float* output, int64_t count) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride) {
    output[i] = 3.0f * input[i];
  }
}

torch::Tensor smoke_cuda(const torch::Tensor& input) {
  const c10::cuda::CUDAGuard guard(input.device());
  auto output = torch::empty_like(input);
  const int64_t count = input.numel();
  if (count == 0) return output;
  const int blocks = static_cast<int>(std::min<int64_t>((count + 255) / 256, 4096));
  const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
  triple_kernel<<<blocks, 256, 0, stream.stream()>>>(
      input.data_ptr<float>(), output.data_ptr<float>(), count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
