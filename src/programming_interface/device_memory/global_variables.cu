#include <cuda_runtime.h>

__constant__ float constData[256];
__device__ float   devData;
__device__ float*  devPointer;

int main() {
  float data[256];
  cudaMemcpyToSymbol(constData, data, sizeof(data));
  cudaMemcpyFromSymbol(data, constData, sizeof(data));

  float value = 3.14f;
  cudaMemcpyToSymbol(devData, &value, sizeof(float));

  float* ptr;
  cudaMalloc(&ptr, 256 * sizeof(float));
  cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));

  cudaFree(ptr);
}
