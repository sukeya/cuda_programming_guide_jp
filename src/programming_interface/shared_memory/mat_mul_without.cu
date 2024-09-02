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
// M(row, col) = *(M.elements + row * M.width + col)
struct Matrix {
    int width;
    int height;
    float* elements;
};

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
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