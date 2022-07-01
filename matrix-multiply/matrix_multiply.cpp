
#include <chrono>
#include <functional>
#include <cstdio>
#include <cassert>
#include <cstdlib>

template <long N>
void assert_arrays_equal(float (&A)[N][N], float (&B)[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (abs(A[i][j] - B[i][j]) > 1e-5)
            {
                printf("A: %f, B:%f\n", A[i][j], B[i][j]);
                exit(-1);
            }
        }
    }
}

template <long N>
void benchmark_matrix_multiply(std::function<void(float (&)[N][N], float (&)[N][N], float (&)[N][N])> matrix_multiply_func,
                               float (&A)[N][N], float (&B)[N][N], float (&C)[N][N], float (&expected)[N][N])
{
    constexpr long FLOPs = 2 * N * N * N;

    auto t1 = std::chrono::high_resolution_clock::now();

    matrix_multiply_func(A, B, C);

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> execution_time = t2 - t1;

    assert_arrays_equal(C, expected);

    double GFLOPS = FLOPs / execution_time.count() * 1e-9;

    printf("FLOPs: %ld\n", FLOPs);
    printf("Execution time: %.2f seconds\n", execution_time.count());
    printf("GFLOPS: %.4f\n", GFLOPS);
}

template <long N>
void standard_matrix_multiply(float (&A)[N][N], float (&B)[N][N], float (&C)[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float acc = 0;
            for (int k = 0; k < N; k++)
            {
                acc += A[i][k] * B[k][j];
            }
            C[i][j] = acc;
        }
    }
}

template <long N, uint BLOCK_SIZE>
void block_matrix_multiply(float (&A)[N][N], float (&B)[N][N], float (&C)[N][N])
{
    // https://csapp.cs.cmu.edu/public/waside/waside-blocking.pdf

    for (int kk = 0; kk < N; kk += BLOCK_SIZE)
    {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE)
        {
            for (int i = 0; i < N; i++)
            {
                for (int j = jj; j < jj + BLOCK_SIZE; j++)
                {
                    float sum = C[i][j];
                    for (int k = kk; k < kk + BLOCK_SIZE; k++)
                    {
                        sum += A[i][k] * B[k][j];
                    }
                    C[i][j] = sum;
                }
            }
        }
    }
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
    benchmark_matrix_multiply<N>(standard_matrix_multiply<N>, A, B, C, expected);

    // FLOPs: 2147483648
    // Execution time: 3.87 seconds
    // GFLOPS: 0.5551
    benchmark_matrix_multiply<N>(block_matrix_multiply<N, 16>, A, B, C, expected);
}
