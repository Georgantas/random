
#include "matrix_multiplier.hpp"

#pragma once

template <long N, int BLOCK_SIZE>
class BlockMatrixMultipler : public IMatrixMultiplier<N>
{
public:
    virtual void multiply(float (&A)[N][N], float (&B)[N][N], float (&C)[N][N]) override
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
};
