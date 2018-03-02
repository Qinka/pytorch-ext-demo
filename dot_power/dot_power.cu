#define _DOT_POWER_C_
#include "dot_power.h"

__global__ static
void kernel_dot_power_forward(const float* base, const float* exponent,
                              float* output,
                              int n) {
  auto index  = blockIdx.x * blockDim.x + threadIdx.x;
  auto stride = blockDim.x * gridDim.x;

  for(auto i = index; i < n; i += stride)
    output[i] = powf(base[i],exponent[i]);
}

__global__ static
void kernel_dot_power_backward(const float* base, const float* exponent,
                               const float* grad_output,
                               float* grad_base, float* grad_exponent,
                               int n) {
  auto index  = blockIdx.x * blockDim.x + threadIdx.x;
  auto stride = blockDim.x * gridDim.x;

  for(auto i = index; i < n; i += stride) {
    grad_base[i] = grad_output[i] * exponent[i] * powf(base[i], exponent[i] - 1);
    grad_exponent[i] = grad_output[i] * logf(exponent[i]) * powf(base[i],exponent[i]);
  }
}

cudaError_t dot_power_forward(const float* base, const float* exponent,
                              float* output,
                              int n, cudaStream_t stream) {
  kernel_dot_power_forward<<<(n-1)/256+1,256,0,stream>>>(base,exponent,output,n);
  return cudaGetLastError();
  //return cudaDeviceSynchronize();
}

cudaError_t dot_power_backward(const float* base, const float* exponent,
                               const float* grad_output,
                               float* grad_base, float* grad_exponent,
                               int n, cudaStream_t stream) {
  kernel_dot_power_backward<<<(n-1)/256+1,256,0,stream>>>(base,exponent,grad_output,
                                                          grad_base,grad_exponent,
                                                          n);
  return cudaGetLastError();
  //return cudaDeviceSynchronize();
}
