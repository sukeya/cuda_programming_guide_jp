#include <cassert>

#include <cuda_runtime.h>

// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  constexpr int    N    = 1e6;
  constexpr size_t size = N * sizeof(float);

  // Allocate input vectors h_A and h_B in host memory
  float* h_A = (float*)std::malloc(size);
  float* h_B = (float*)std::malloc(size);
  float* h_C = (float*)std::malloc(size);

  // Initialize input vectors
  for (int i = 0; i < N; ++i) {
    h_A[i] = i;
  }

  for (int i = 0; i < N; ++i) {
    h_B[i] = 2 * i;
  }

  // Allocate vectors in device memory
  float* d_A;
  cudaMalloc(&d_A, size);
  float* d_B;
  cudaMalloc(&d_B, size);
  float* d_C;
  cudaMalloc(&d_C, size);

  // Copy vectors from host memory to device memory
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Invoke kernel
  int threadsPerBlock = 256;
  int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
  VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

  // Copy result from device memory to host memory
  // h_C contains the result in host memory
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // Check result
  for (int i = 0; i < N; ++i) {
    assert(h_C[i] == 3 * i);
  }

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free host memory
  std::free(h_A);
  std::free(h_B);
  std::free(h_C);
}
