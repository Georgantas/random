
#include <cublas_matrix_multiplier.hpp>
#include <cuda_runtime.h>
#include <cassert>
#include <cublas_v2.h>
#include <cstdio>

template <long N>
CublasMatrixMultiplier<N>::CublasMatrixMultiplier()
{
    cublasStatus_t cublasStatus = cublasCreate(&cublasHandle);
    assert(cublasStatus == CUBLAS_STATUS_SUCCESS);
}

template <long N>
CublasMatrixMultiplier<N>::~CublasMatrixMultiplier()
{
    cublasDestroy(cublasHandle);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

template <long N>
void CublasMatrixMultiplier<N>::multiply(float (&A)[N][N], float (&B)[N][N], float (&C)[N][N])
{
    cudaError cudaStatus = cudaMalloc(&A_d, N * N * sizeof(*A_d));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&B_d, N * N * sizeof(*B_d));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&C_d, N * N * sizeof(*C_d));
    assert(cudaStatus == cudaSuccess);

    cudaStatus = cudaMemcpy2D(A_d, N * sizeof(float), A, N * sizeof(float), N * sizeof(float), N, cudaMemcpyHostToDevice);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMemcpy2D(B_d, N * sizeof(float), B, N * sizeof(float), N * sizeof(float), N, cudaMemcpyHostToDevice);
    assert(cudaStatus == cudaSuccess);

    const float alpha = 1;
    const float beta = 0;

    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, B_d, N, A_d, N, &beta, C_d, N);

    cudaStatus = cudaMemcpy2D(C, N * sizeof(float), C_d, N * sizeof(float), N * sizeof(float), N, cudaMemcpyDeviceToHost);
    assert(cudaStatus == cudaSuccess);
}

// explicit instantion: https://docs.microsoft.com/en-us/cpp/cpp/explicit-instantiation?view=msvc-170
template class CublasMatrixMultiplier<1024>;
