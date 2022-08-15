
## Completed
### StandardMatrixMultiplier
#### Explanation
Standard triple for-loop using the CPU.
#### Performance
- **On Dell XPS 13:** FLOPs: 2147483648; Execution time: 5.25 seconds; GFLOPS: 0.4090;
---
### BlockMatrixMultiplier
#### Explanation
Using the CPU with blocking for more temporal and spatial locality. Leverages the L1 cache. More details [here](https://csapp.cs.cmu.edu/public/waside/waside-blocking.pdf).
#### Performance
- **On Dell XPS 13:** FLOPs: 2147483648; Execution time: 3.35 seconds; GFLOPS: 0.6402;
---
### OpenCLMatrixMultiplier
#### Explanation
Using the GPU with a basic OpenCL kernel.
#### Performance
- **Platform: [NVIDIA TITAN Xp](https://vast.ai) / Device: NVIDIA CUDA:** FLOPs: 2147483648; Execution time: 0.04 seconds; GFLOPS: 51.8588;
- **Platform: Intel(R) Core(TM) i7-10710U CPU @ 1.10GHz / Device: Intel(R) OpenCL:** FLOPs: 2147483648; Execution time: 0.10 seconds; GFLOPS: 20.7495;
---
### CublasMatrixMultiplier
#### Explanation
Use cuBLAS.
#### Performance
- **Platform: [Tesla K80](https://vast.ai):** FLOPs: 2147483648; Execution time: 0.01 seconds; GFLOPS: 304.7607;
- **Platform: [RTX 3090](https://vast.ai):** FLOPs: 2147483648; Execution time: 0.00 seconds; GFLOPS: 773.4415;
---
### CudaBlockMatrixMultipler
#### Explanation
Load blocks into GPU shared memory to reduce global memory accesses. Explained in detail [here](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory).

![image](https://user-images.githubusercontent.com/18753033/184559807-d2f5da04-7492-4e6c-92c4-bbfb4e14d4b0.png)

#### Performance
- **Platform: [RTX 3090](https://vast.ai):** FLOPs: 2147483648; Execution time: 0.00 seconds; GFLOPS: 951.0809;
---
## TODO
### TransposedBlockMatrixMultiplier
#### Explanation
Similar to BlockMatrixMultiplier, but load the matrix "B" to memory transposed and use SIMD instructions to perform the block dot products.
#### Performance
- TODO
---
### NumpyMatrixMultiplier
#### Explanation
Multiply with a matrix in python using numpy for comparison.
#### Performance
- TODO
---
### GPGPUMatrixMultiplier
#### Explanation
Multiply with a GPU Shader.
#### Performance
- TODO
---

