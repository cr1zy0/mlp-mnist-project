#include "../include/matrix.h"
#include <assert.h>
#include <stdlib.h>

Matrix createMatrix(int rows, int cols) {
  Matrix matrix = {0, 0, NULL};
  if (rows <= 0 || cols <= 0) {
    return matrix;
  }
  matrix.data = malloc(sizeof(double) * rows * cols);
  if (matrix.data == NULL) {
    return matrix;
  }
  matrix.cols = cols;
  matrix.rows = rows;
  return matrix;
}
void freeMatrix(Matrix *matrix) {
  free(matrix->data);
  matrix->rows = 0;
  matrix->cols = 0;
  matrix->data = NULL;
}

static void checkBoundaries(const Matrix *matrix, int row, int col) {
  assert(matrix != NULL);
  assert(matrix->data != NULL);
  assert(row >= 0 && row < matrix->rows);
  assert(col >= 0 && col < matrix->cols);
}
double matrix_get(const Matrix *matrix, int row, int col) {
  checkBoundaries(matrix, row, col);
  return matrix->data[matrix->cols * row + col];
}
void matrix_set(Matrix *matrix, int row, int col, double val) {
  checkBoundaries(matrix, row, col);
  matrix->data[matrix->cols * row + col] = val;
}
double *matrix_multiply_vector(const Matrix *matrix, const double *vector,
                               int vector_size) {
  assert(matrix != NULL);
  assert(vector != NULL);
  assert(vector_size == matrix->cols);
  assert(matrix->data != NULL);
  double *res = calloc(matrix->rows, sizeof(double));
  if (res == NULL) {
    return NULL;
  }
  for (int i = 0; i < matrix->rows; i++) {
    for (int j = 0; j < matrix->cols; j++) {
      res[i] += matrix_get(matrix, i, j) * vector[j];
    }
  }
  return res;
}
