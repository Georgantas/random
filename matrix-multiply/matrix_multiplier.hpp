
#pragma once

template<long N>
class IMatrixMultiplier { 
    public:
        virtual void multiply(float (&)[N][N], float (&)[N][N], float (&)[N][N]) = 0;
};

