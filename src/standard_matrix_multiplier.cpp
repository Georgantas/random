
#include <standard_matrix_multiplier.hpp>

template <long N>
void StandardMatrixMultipler<N>::multiply(float (&A)[N][N], float (&B)[N][N], float (&C)[N][N])
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

template class StandardMatrixMultipler<1024>;
