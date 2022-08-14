
#include <matrix_multiplier.hpp>

#pragma once

template <long N, int BLOCK_SIZE>
class BlockMatrixMultiplier : public IMatrixMultiplier<N>
{
public:
    virtual void multiply(float (&A)[N][N], float (&B)[N][N], float (&C)[N][N]) override;
};
