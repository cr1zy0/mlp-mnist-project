#include "../include/matrix.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  Matrix matrix = createMatrix(2, 3);
  assert(matrix.data != NULL);

  printf("Matrix: %d x %d\n", matrix.rows, matrix.cols);

  for (int i = 0; i < matrix.rows; i++) {
    for (int j = 0; j < matrix.cols; j++) {
      matrix_set(&matrix, i, j, i * matrix.cols + j + 1);
      printf("%.1f ", matrix_get(&matrix, i, j));
    }
    printf("\n");
  }

  double vector[] = {10, 20, 30};

  printf("\nMatrix x vector:\n");

  double *result = matrix_multiply_vector(&matrix, vector, 3);
  assert(result != NULL);

  for (int i = 0; i < matrix.rows; i++) {
    printf("%.1f\n", result[i]);
  }

  assert(result[0] == 140.0);
  assert(result[1] == 320.0);

  free(result);
  freeMatrix(&matrix);

  assert(matrix.data == NULL);
  assert(matrix.rows == 0);
  assert(matrix.cols == 0);

  printf("\nAll tests passed!\n");

  return 0;
}
