
void kernel matrix_multiply(global const float *A, global const float *B,
                            global float *C, int N) {
  size_t row = get_global_id(0);
  size_t column = get_global_id(1);

  float acc = 0.f;
  for (size_t k = 0; k < N; k++) {
    acc += A[row * N + k] * B[k * N + column];
  }

  C[row * N + column] = acc;
}
