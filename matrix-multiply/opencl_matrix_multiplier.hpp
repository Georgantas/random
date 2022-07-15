
#include "matrix_multiplier.hpp"
#include <CL/cl.hpp>
#include <cstdio>

#pragma once

template <long N>
class OpenCLMatrixMultiplier : public IMatrixMultiplier<N>
{
public:
    virtual void multiply(float (&A)[N][N], float (&B)[N][N], float (&C)[N][N])
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
        matrix_multiply(cl::EnqueueArgs(queue, global), A_d, B_d, C_d, (int)N).wait();
        queue.enqueueReadBuffer(C_d, CL_TRUE, 0, sizeof(float) * N * N, C);
    }
};
