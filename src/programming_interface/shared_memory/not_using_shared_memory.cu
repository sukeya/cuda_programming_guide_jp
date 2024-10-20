#include <cassert>
#include <memory>

// CUDA runtime
#include <cuda_runtime.h>

// Thread block size
#define BLOCK_SIZE 16

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
struct Matrix {
  int    width;
  int    height;
  float* elements;
};

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C) {
  // Load A and B to device memory
  Matrix d_A;
  d_A.width   = A.width;
  d_A.height  = A.height;
  size_t size = A.width * A.height * sizeof(float);
  cudaMalloc(&d_A.elements, size);
  cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
  Matrix d_B;
  d_B.width  = B.width;
  d_B.height = B.height;
  size       = B.width * B.height * sizeof(float);
  cudaMalloc(&d_B.elements, size);
  cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

  // Allocate C in device memory
  Matrix d_C;
  d_C.width  = C.width;
  d_C.height = C.height;
  size       = C.width * C.height * sizeof(float);
  cudaMalloc(&d_C.elements, size);

  // Invoke kernel
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
  MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

  // Read C from device memory
  cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_A.elements);
  cudaFree(d_B.elements);
  cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
// Calculate C = A * B
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
  // Each thread computes one element of C
  // by accumulating results into Cvalue
  float Cvalue = 0;
  int   row    = blockIdx.y * blockDim.y + threadIdx.y;
  int   col    = blockIdx.x * blockDim.x + threadIdx.x;
  for (int e = 0; e < A.width; ++e)
    Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
  C.elements[row * C.width + col] = Cvalue;
}

int main(int argc, char** argv) {
  int cols = 3 * BLOCK_SIZE;
  int rows = 3 * BLOCK_SIZE;

  auto A = std::make_unique<float[]>(rows * cols);
  auto B = std::make_unique<float[]>(rows * cols);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      A[i * cols + j] = i * j;
    }
  }

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      B[i * cols + j] = i + j;
    }
  }

  auto A_cuda = Matrix{.width = cols, .height = rows, .elements = A.get()};
  auto B_cuda = Matrix{.width = cols, .height = rows, .elements = B.get()};

  auto C      = std::make_unique<float[]>(rows * cols);
  auto C_cuda = Matrix{.width = cols, .height = rows, .elements = C.get()};

  MatMul(A_cuda, B_cuda, C_cuda);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      auto answer = 0.0f;
      for (int k = 0; k < rows; ++k) {
        answer += (i * k) * (k + j);
      }
      assert(C[i * cols + j] == answer);
    }
  }

  return 0;
}