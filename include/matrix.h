typedef struct {
  int rows;
  int cols;
  double *data;
} Matrix;

Matrix createMatrix(int rows, int cols);
void freeMatrix(Matrix *matrix);
double matrix_get(const Matrix *matrix, int row, int col);
void matrix_set(Matrix *matrix, int row, int col, double val);
double *matrix_multiply_vector(const Matrix *matrix, const double *vector,
                               int vector_size);
