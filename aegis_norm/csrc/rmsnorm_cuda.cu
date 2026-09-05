#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

template <typename scalar_t>
__global__ void rmsnorm_kernel(const scalar_t* x, const scalar_t* weight,
                               scalar_t* output, int64_t width, float eps) {
  __shared__ float warp_sums[8];
  __shared__ float inverse_rms;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int64_t base = static_cast<int64_t>(blockIdx.x) * width;

  float local_sum = 0.0f;
  for (int64_t i = threadIdx.x; i < width; i += blockDim.x) {
    const float value = static_cast<float>(x[base + i]);
    local_sum += value * value;
  }
  local_sum = warp_sum(local_sum);
  if (lane == 0) warp_sums[warp] = local_sum;
  __syncthreads();

  if (warp == 0) {
    // All 32 lanes participate; only eight hold nonzero warp partials.
    float total = lane < 8 ? warp_sums[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) inverse_rms = rsqrtf(total / static_cast<float>(width) + eps);
  }
  __syncthreads();

  for (int64_t i = threadIdx.x; i < width; i += blockDim.x) {
    // Preserve rounding to input dtype BEFORE multiplication by gamma.
    const scalar_t normalized = static_cast<scalar_t>(static_cast<float>(x[base + i]) * inverse_rms);
    output[base + i] = static_cast<scalar_t>(
        static_cast<float>(normalized) * static_cast<float>(weight[i]));
  }
}

at::Tensor rmsnorm_launch(const at::Tensor& x, const at::Tensor& weight, float eps) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto output = at::empty(x.sizes(), x.options());
  const int64_t width = x.size(-1);
  const int64_t rows = x.numel() / width;
  if (rows == 0) return output;
  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  if (x.scalar_type() == at::kHalf) {
    rmsnorm_kernel<at::Half><<<static_cast<unsigned int>(rows), 256, 0, stream.stream()>>>(
        x.data_ptr<at::Half>(), weight.data_ptr<at::Half>(), output.data_ptr<at::Half>(), width, eps);
  } else {
    rmsnorm_kernel<float><<<static_cast<unsigned int>(rows), 256, 0, stream.stream()>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), width, eps);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
