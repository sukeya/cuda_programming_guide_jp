#include <chrono>
#include <iostream>
#include <memory>

// CUDA runtime
#include <cuda_runtime.h>

#include "argparse/argparse.hpp"
#include "Eigen/Dense"

// Thread block size
#define BLOCK_SIZE 16


// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
struct Matrix{
    int width;
    int height;
    int stride;
    float* elements;
};


// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}


// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}


// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}


// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);


// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = d_B.stride = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
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
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            Cvalue += As[row][e] * Bs[e][col];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}


bool is_equal(float lhd, float rhd) {
    if (std::abs(lhd) < 1 || std::abs(rhd) < 1) {
        return std::abs(lhd - rhd) < 1e-4;
    } else {
        return std::abs(2 * (lhd - rhd) / (lhd + rhd)) < 1e-4;
    }
}


int main(int argc, char** argv) {
    using MyMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    argparse::ArgumentParser program("mat_mul_without");

    program.add_argument("--cols")
        .required()
        .help("Column size of the matrix")
        .scan<'i', int>();

    program.add_argument("--rows")
        .required()
        .help("Row size of the matrix")
        .scan<'i', int>();

    program.add_argument("--help")
        .help("Print help message");

    program.add_argument("--no-check")
        .help("Print version information")
        .default_value(false)
        .implicit_value(true);

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cout << program;
        return 1;
    }

    auto cols = program.get<int>("--cols");
    auto rows = program.get<int>("--rows");

    if (cols % BLOCK_SIZE != 0 || rows % BLOCK_SIZE != 0) {
        std::cerr << "Matrix dimensions must be multiples of " << BLOCK_SIZE << std::endl;
        return 1;
    }

    MyMatrix A = MyMatrix::Random(rows, cols);
    MyMatrix B = MyMatrix::Random(rows, cols);

    auto A_cuda = Matrix{.width = A.cols(), .height = A.rows(), .elements = A.data()};
    auto B_cuda = Matrix{.width = B.cols(), .height = B.rows(), .elements = B.data()};

    auto C_cuda_elem_ptr = std::make_unique<float[]>(A.rows() * B.cols());
    auto C_cuda = Matrix{.width = B.cols(), .height = A.rows(), .elements = C_cuda_elem_ptr.get()};

    auto start = std::chrono::system_clock::now();

    MatMul(A_cuda, B_cuda, C_cuda);

    auto end = std::chrono::system_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    if (program["--no-check"] == false) {
        MyMatrix C = A * B;
        if (C_cuda.width != C.cols() || C_cuda.height != C.rows()) {
            std::cerr << "Dimensions do not match" << std::endl;
            return 1;
        }
        for (int i = 0; i < C.rows(); ++i) {
            for (int j = 0; j < C.cols(); ++j) {
                if (not is_equal(C(i, j), C_cuda.elements[i * C.cols() + j])) {
                    std::cerr << "Mismatch at (" << i << ", " << j << "): " << C(i, j) << " != " << C_cuda.elements[i * C.cols() + j] << std::endl;
                    return 1;
                }
            }
        }
    }
    std::cout << "Success" << std::endl;
    return 0;
}
