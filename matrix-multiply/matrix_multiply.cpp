
#include <chrono>
#include <functional>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <streambuf>
#include <CL/cl.hpp>

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
void benchmark_matrix_multiply(std::function<void(float (&)[N][N], float (&)[N][N], float (&)[N][N])> matrix_multiply_func,
                               float (&A)[N][N], float (&B)[N][N], float (&C)[N][N], float (&expected)[N][N])
{
    constexpr long FLOPs = 2 * N * N * N;

    clear_matrix(C);

    auto t1 = std::chrono::high_resolution_clock::now();

    matrix_multiply_func(A, B, C);

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> execution_time = t2 - t1;

    assert_arrays_equal(C, expected);

    double GFLOPS = FLOPs / execution_time.count() * 1e-9;

    printf("FLOPs: %ld\n", FLOPs);
    printf("Execution time: %.2f seconds\n", execution_time.count());
    printf("GFLOPS: %.4f\n\n", GFLOPS);
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

template <long N>
void benchmark_opencl_matrix_multiply(float (&A)[N][N], float (&B)[N][N], float (&C)[N][N]) //, float (&expected)[N][N])
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    assert(platforms.size() == 1);
    cl::Platform platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    assert(devices.size() == 1);
    cl::Device device = devices.front();

    printf("Platform: %s\nDevice: %s\n\n", device.getInfo<CL_DEVICE_NAME>().c_str(), platform.getInfo<CL_PLATFORM_NAME>().c_str());

    cl::Context context({device});
    cl::Buffer A_d(context, CL_MEM_READ_ONLY, sizeof(float) * N * N);
    cl::Buffer B_d(context, CL_MEM_READ_ONLY, sizeof(float) * N * N);
    cl::Buffer C_d(context, CL_MEM_WRITE_ONLY, sizeof(float) * N * N);

    cl::CommandQueue queue(context, device);

    std::ifstream kernel_code_stream("matrix_multiply_kernel.cl");
    std::string kernel_code((std::istreambuf_iterator<char>(kernel_code_stream)),
                            std::istreambuf_iterator<char>());

    cl::Program::Sources sources;
    sources.emplace_back(kernel_code.c_str(), kernel_code.length());

    cl::Program program(context, sources);

    assert(program.build({device}) == CL_SUCCESS);

    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int> matrix_multiply(cl::Kernel(program, "matrix_multiply"));

    cl::NDRange global(N, N);
    queue.enqueueWriteBuffer(A_d, CL_TRUE, 0, sizeof(float) * N * N, A);
    queue.enqueueWriteBuffer(B_d, CL_TRUE, 0, sizeof(float) * N * N, B);
    matrix_multiply(cl::EnqueueArgs(queue, global), A_d, B_d, C_d, (int) N).wait();
    queue.enqueueReadBuffer(C_d, CL_TRUE, 0, sizeof(float) * N * N, C);
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

    // FLOPs: 2147483648
    // Execution time: 0.33 seconds
    // GFLOPS: 6.4878
    benchmark_matrix_multiply<N>(benchmark_opencl_matrix_multiply<N>, A, B, C, expected);
}
