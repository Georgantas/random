
#include <opencl_matrix_multiplier.hpp>
#include <CL/cl.hpp>
#include <cstdio>
#include <memory>
#include <cassert>
#include <fstream>

template <long N>
OpenCLMatrixMultiplier<N>::OpenCLMatrixMultiplier()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    assert(platforms.size() > 0);
    cl::Platform platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    assert(devices.size() > 0);
    cl::Device device = devices.front();

    printf("Platform: %s\nDevice: %s\n\n", device.getInfo<CL_DEVICE_NAME>().c_str(), platform.getInfo<CL_PLATFORM_NAME>().c_str());

    cl::Context context({device});
    A_d = std::make_unique<cl::Buffer>(context, CL_MEM_READ_ONLY, sizeof(float) * N * N);
    B_d = std::make_unique<cl::Buffer>(context, CL_MEM_READ_ONLY, sizeof(float) * N * N);
    C_d = std::make_unique<cl::Buffer>(context, CL_MEM_WRITE_ONLY, sizeof(float) * N * N);

    queue = std::make_unique<cl::CommandQueue>(context, device);

    std::ifstream kernel_code_stream("matrix_multiply_kernel.cl");
    std::string kernel_code((std::istreambuf_iterator<char>(kernel_code_stream)),
                            std::istreambuf_iterator<char>());

    cl::Program::Sources sources;
    sources.emplace_back(kernel_code.c_str(), kernel_code.length());

    cl::Program program(context, sources);

    assert(program.build({device}) == CL_SUCCESS);

    matrix_multiply = std::make_unique<cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int>>(cl::Kernel(program, "matrix_multiply"));

    global = std::make_unique<cl::NDRange>(N, N);
}

template <long N>
void OpenCLMatrixMultiplier<N>::multiply(float (&A)[N][N], float (&B)[N][N], float (&C)[N][N])
{
    queue->enqueueWriteBuffer(*A_d, CL_TRUE, 0, sizeof(float) * N * N, A);
    queue->enqueueWriteBuffer(*B_d, CL_TRUE, 0, sizeof(float) * N * N, B);
    (*matrix_multiply)(cl::EnqueueArgs(*queue, *global), *A_d, *B_d, *C_d, (int)N).wait();
    queue->enqueueReadBuffer(*C_d, CL_TRUE, 0, sizeof(float) * N * N, C);
}

template class OpenCLMatrixMultiplier<1024>;