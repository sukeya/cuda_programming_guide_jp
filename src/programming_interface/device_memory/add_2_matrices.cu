#include <cassert>

#include <cuda_runtime.h>

__device__ float& at(float* M, std::size_t pitch, int row, int column) {
  return *((float*)((char*)M + row * pitch) + column);
}

__global__ void MatrixAdd(
    float*      A,
    std::size_t pitch_A,
    float*      B,
    std::size_t pitch_B,
    float*      C,
    std::size_t pitch_C,
    int         row,
    int         column
) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i < row and j < column) {
    at(C, pitch_C, i, j) = at(A, pitch_A, i, j) + at(B, pitch_B, i, j);
  }
}

template <class T>
void AllocateMatrix(T** dev_ptr, std::size_t* pitch, int row, int column) {
  auto e = cudaMallocPitch(dev_ptr, pitch, sizeof(T) * column, row);
  assert(e == cudaSuccess);
}

void FreeMatrix(void* dev_ptr) {
  auto e = cudaFree(dev_ptr);
  assert(e == cudaSuccess);
}

template <class T>
void CopyMatrixHostToDevice(T* dst, size_t dpitch, const T* src, int row, int column) {
  auto e = cudaMemcpy2D(
      dst,
      dpitch,
      src,
      sizeof(T) * column,
      sizeof(T) * column,
      row,
      cudaMemcpyHostToDevice
  );
  assert(e == cudaSuccess);
}

template <class T>
void CopyMatrixDeviceToHost(T* dst, const T* src, size_t dpitch, int row, int column) {
  auto e = cudaMemcpy2D(
      dst,
      sizeof(T) * column,
      src,
      dpitch,
      sizeof(T) * column,
      row,
      cudaMemcpyDeviceToHost
  );
  assert(e == cudaSuccess);
}

int main() {
  constexpr int         row    = 64;
  constexpr int         column = 64;
  constexpr std::size_t size   = sizeof(float) * row * column;

  // Allocate input vectors h_A and h_B in host memory
  float* h_A = (float*)std::malloc(size);
  assert(h_A != nullptr);
  float* h_B = (float*)std::malloc(size);
  assert(h_B != nullptr);
  float* h_C = (float*)std::malloc(size);
  assert(h_C != nullptr);

  // Initialize input vectors
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < column; ++j) {
      h_A[i * column + j] = i * j;
    }
  }

  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < column; ++j) {
      h_B[i * column + j] = 2 * i * j;
    }
  }

  float* d_A;
  float* d_B;
  float* d_C;

  std::size_t pitch_A;
  std::size_t pitch_B;
  std::size_t pitch_C;

  AllocateMatrix(&d_A, &pitch_A, row, column);
  AllocateMatrix(&d_B, &pitch_B, row, column);
  AllocateMatrix(&d_C, &pitch_C, row, column);

  CopyMatrixHostToDevice(d_A, pitch_A, h_A, row, column);
  CopyMatrixHostToDevice(d_B, pitch_B, h_B, row, column);

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid(row / threadsPerBlock.x, column / threadsPerBlock.y);
  MatrixAdd<<<blocksPerGrid, threadsPerBlock>>>(
      d_A,
      pitch_A,
      d_B,
      pitch_B,
      d_C,
      pitch_C,
      row,
      column
  );

  CopyMatrixDeviceToHost(h_C, d_C, pitch_C, row, column);

  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < column; ++j) {
      assert(h_C[i * column + j] == 3 * i * j);
    }
  }

  FreeMatrix(d_A);
  FreeMatrix(d_B);
  FreeMatrix(d_C);

  std::free(h_A);
  std::free(h_B);
  std::free(h_C);
}
