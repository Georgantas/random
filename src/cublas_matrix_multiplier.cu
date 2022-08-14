
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

    cudaStatus = cudaMemcpy2D(A_d, 0, A, 0, N, N, cudaMemcpyHostToDevice);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMemcpy2D(B_d, 0, B, 0, N, N, cudaMemcpyHostToDevice);
    assert(cudaStatus == cudaSuccess);

    const float alpha = 1;
    const float beta = 0;

    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A_d, N, B_d, N, &beta, C_d, N);

    cudaMemcpy2D(C, 0, C_d, 0, N, N, cudaMemcpyDeviceToHost);
}

// explicit instantion: https://docs.microsoft.com/en-us/cpp/cpp/explicit-instantiation?view=msvc-170
template class CublasMatrixMultiplier<1024>;
