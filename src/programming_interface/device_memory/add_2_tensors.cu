#include <cassert>

#include <cuda_runtime.h>

__device__ float& at(cudaPitchedPtr devPitchedPtr,
                         int height, int x, int y, int z) {
  char* devPtr = (char*)(devPitchedPtr.ptr);
  size_t pitch = devPitchedPtr.pitch;
  size_t slicePitch = pitch * height;
  float* row = (float*)(devPtr + z * slicePitch + y * pitch);
  return row[x];
}

__global__ void TensorAdd(
    cudaPitchedPtr      A,
    cudaPitchedPtr      B,
    cudaPitchedPtr      C,
    int         width,
    int         height,
    int depth
) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i < width and j < height and k < depth) {
    at(C, height, i, j, k) = at(A, height, i, j, k) + at(B, height, i, j, k);
  }
}

void AllocateTensor(cudaPitchedPtr* devPitchedPtr, int width, int height, int depth) {
  cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
  auto e = cudaMalloc3D(devPitchedPtr, extent);
  assert(e == cudaSuccess);
}

void FreeTensor(void* dev_ptr) {
  auto e = cudaFree(dev_ptr);
  assert(e == cudaSuccess);
}

void CopyTensor(cudaPitchedPtr dst, cudaPitchedPtr src, int width, int height, int depth) {
  cudaMemcpy3DParms myParms = {0};
  myParms.srcPtr = src;
  myParms.dstPtr = dst;
  myParms.extent = make_cudaExtent(width * sizeof(float), height, depth);
  myParms.kind = cudaMemcpyDefault;

  auto e = cudaMemcpy3D(&myParms);
  assert(e == cudaSuccess);
}

void InitTensor(cudaPitchedPtr& T, int width, int height, int depth) {
  std::size_t size   = sizeof(float) * width * height * depth;

  // Allocate input vectors h_A and h_B in host memory
  T.ptr = (float*)std::malloc(size);
  assert(T.ptr != nullptr);

  T.pitch = sizeof(float) * width;

  T.xsize = width;
  T.ysize = height;
}

int main() {
  constexpr int         width    = 64;
  constexpr int         height = 64;
  constexpr int         depth = 64;

  cudaPitchedPtr h_A;
  cudaPitchedPtr h_B;
  cudaPitchedPtr h_C;

  InitTensor(h_A, width, height, depth);
  InitTensor(h_B, width, height, depth);
  InitTensor(h_C, width, height, depth);

  // Initialize input vectors
  for (int k = 0; k < depth; ++k) {
    for (int j = 0; j < height; ++j) {
      for (int i = 0; i < width; ++i) {
        float* arr = (float*)h_A.ptr;
        arr[i + j * width + k * width * height] = i * j * k;
      }
    }
  }

  for (int k = 0; k < depth; ++k) {
    for (int j = 0; j < height; ++j) {
      for (int i = 0; i < width; ++i) {
        float* arr = (float*)h_B.ptr;
        arr[i + j * width + k * width * height] = 2 * i * j * k;
      }
    }
  }

  cudaPitchedPtr d_A;
  cudaPitchedPtr d_B;
  cudaPitchedPtr d_C;

  AllocateTensor(&d_A, width, height, depth);
  AllocateTensor(&d_B, width, height, depth);
  AllocateTensor(&d_C, width, height, depth);

  CopyTensor(d_A, h_A, width, height, depth);
  CopyTensor(d_B, h_B, width, height, depth);

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid(width / threadsPerBlock.x, height / threadsPerBlock.y, depth / threadsPerBlock.z);
  TensorAdd<<<blocksPerGrid, threadsPerBlock>>>(
      d_A,
      d_B,
      d_C,
      width,
      height,
      depth
  );

  CopyTensor(h_C, d_C, width, height, depth);

  for (int k = 0; k < depth; ++k) {
    for (int j = 0; j < height; ++j) {
      for (int i = 0; i < width; ++i) {
        float* arr = (float*)h_C.ptr;
        assert(arr[i + j * width + k * width * height] == 3 * i * j * k);
      }
    }
  }

  FreeTensor(d_A.ptr);
  FreeTensor(d_B.ptr);
  FreeTensor(d_C.ptr);

  std::free(h_A.ptr);
  std::free(h_B.ptr);
  std::free(h_C.ptr);
}
