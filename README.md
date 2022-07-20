| Method | Explanation | Result |
|---|---|---|
| StandardMatrixMultiplier | Standard triple for-loop using the CPU.  | <li> **On Dell XPS 13:** FLOPs: 2147483648; Execution time: 5.25 seconds; GFLOPS: 0.4090;</li> |
| BlockMatrixMultiplier | Using the CPU with blocking for more temporal and spatial locality. Leverages the L1 cache. More details [here](https://csapp.cs.cmu.edu/public/waside/waside-blocking.pdf). | <li> **On Dell XPS 13:** FLOPs: 2147483648; Execution time: 3.35 seconds; GFLOPS: 0.6402; </li> |
| OpenCLMatrixMultiplier | Using the GPU with a basic OpenCL kernel. | <li>**Platform: [NVIDIA TITAN Xp](https://vast.ai) / Device: NVIDIA CUDA:** FLOPs: 2147483648; Execution time: 0.04 seconds; GFLOPS: 51.8588;</li> <li>**Platform: Intel(R) Core(TM) i7-10710U CPU @ 1.10GHz / Device: Intel(R) OpenCL:** FLOPs: 2147483648; Execution time: 0.10 seconds; GFLOPS: 20.7495;</li> |
| BlockOpenCLMatrixMultiplier (todo) | Similar to OpenCLMatrixMultiplier, but load blocks into local memory. | |
| TransposedBlockMatrixMultiplier (todo) | Similar to BlockMatrixMultiplier, but load the matrix "B" to memory transposed and use SIMD instructions to perform the block dot products. |  |
| NumpyMatrixMultiplier (todo) | Multiply with a matrix in python using numpy for comparison. | |
| CublasMatrixMultiplier (todo) | Use cuBLAS. | |
| GPGPUMatrixMultiplier (todo) | Multiply with a GPU Shader. | |
