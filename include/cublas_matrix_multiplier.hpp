
#include <matrix_multiplier.hpp>
#include <cublas_v2.h>

#pragma once

template <long N>
class CublasMatrixMultiplier : public IMatrixMultiplier<N>
{
public:
    CublasMatrixMultiplier();
    ~CublasMatrixMultiplier();
    virtual void multiply(float (&)[N][N], float (&)[N][N], float (&)[N][N]) override;

private:
    cublasHandle_t cublasHandle;
    float* A_d;
    float* B_d;
    float* C_d;
};
