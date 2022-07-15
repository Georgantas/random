
#include <chrono>
#include <functional>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <streambuf>
#include "matrix_multiplier.hpp"
#include "standard_matrix_multiplier.hpp"
#include "block_matrix_multipler.hpp"
#include "opencl_matrix_multiplier.hpp"

template <long N>
void assert_arrays_equal(float (&actual)[N][N], float (&expected)[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (abs(actual[i][j] - expected[i][j]) > 1e-3)
            {
                printf("Array mismatch:\nActual: %f, Expected:%f\n", actual[i][j], expected[i][j]);
                exit(-1);
            }
        }
    }
}

template <long N>
void clear_matrix(float (&C)[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i][j] = 0;
        }
    }
}

template <long N>
void benchmark_matrix_multiply(IMatrixMultiplier<N> &matrix_multiplier,
                               float (&A)[N][N], float (&B)[N][N], float (&C)[N][N], float (&expected)[N][N])
{
    constexpr long FLOPs = 2 * N * N * N;

    clear_matrix(C);

    auto t1 = std::chrono::high_resolution_clock::now();

    matrix_multiplier.multiply(A, B, C);

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> execution_time = t2 - t1;

    assert_arrays_equal(C, expected);

    double GFLOPS = FLOPs / execution_time.count() * 1e-9;

    printf("FLOPs: %ld\n", FLOPs);
    printf("Execution time: %.2f seconds\n", execution_time.count());
    printf("GFLOPS: %.4f\n\n", GFLOPS);
}

#define N 1024

float A[N][N];
float B[N][N];
float C[N][N];
float expected[N][N];

int main()
{
    FILE *file = fopen("1024_matrix_data", "rb");
    assert(file != NULL);

    fread(A, sizeof(float), N * N, file);
    fread(B, sizeof(float), N * N, file);
    fread(expected, sizeof(float), N * N, file);

    fclose(file);

    // FLOPs: 2147483648
    // Execution time: 5.25 seconds
    // GFLOPS: 0.4090
    StandardMatrixMultipler<N> standard_matrix_multiplier;
    benchmark_matrix_multiply<N>(standard_matrix_multiplier, A, B, C, expected);

    // FLOPs: 2147483648
    // Execution time: 3.35 seconds
    // GFLOPS: 0.6402
    BlockMatrixMultipler<N, 16> block_matrix_multiplier;
    benchmark_matrix_multiply<N>(block_matrix_multiplier, A, B, C, expected);

    // FLOPs: 2147483648
    // Execution time: 0.10 seconds
    // GFLOPS: 20.7495
    OpenCLMatrixMultiplier<N> opencl_matrix_multiplier;
    benchmark_matrix_multiply<N>(opencl_matrix_multiplier, A, B, C, expected);
}
