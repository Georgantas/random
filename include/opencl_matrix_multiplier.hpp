
#include <matrix_multiplier.hpp>
#include <memory>
#include <CL/cl.hpp>

#pragma once

template <long N>
class OpenCLMatrixMultiplier : public IMatrixMultiplier<N>
{
public:
    OpenCLMatrixMultiplier();
    virtual void multiply(float (&A)[N][N], float (&B)[N][N], float (&C)[N][N]) override;

private:
    std::unique_ptr<cl::Buffer> A_d;
    std::unique_ptr<cl::Buffer> B_d;
    std::unique_ptr<cl::Buffer> C_d;
    std::unique_ptr<cl::CommandQueue> queue;
    std::unique_ptr<cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int>> matrix_multiply;
    std::unique_ptr<cl::NDRange> global;
};
