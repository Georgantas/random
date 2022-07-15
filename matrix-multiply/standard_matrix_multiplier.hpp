
#include "matrix_multiplier.hpp"

#pragma once

template <long N>
class StandardMatrixMultipler : public IMatrixMultiplier<N>
{
public:
    virtual void multiply(float (&A)[N][N], float (&B)[N][N], float (&C)[N][N]) override
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
};
