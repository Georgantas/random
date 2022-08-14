
#include <matrix_multiplier.hpp>

#pragma once

// Code adapted from example at: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
struct Matrix
{
    int width;
    int height;
    int stride;
    float *elements;
};

template <long N>
class CudaBlockMatrixMultipler : public IMatrixMultiplier<N>
{
public:
    ~CudaBlockMatrixMultipler();
    virtual void multiply(float (&A)[N][N], float (&B)[N][N], float (&C)[N][N]) override;

private:
    Matrix d_A;
    Matrix d_B;
    Matrix d_C;
};

