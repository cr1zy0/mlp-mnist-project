#include "../include/network.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>

void freeForwardCache(ForwardCache *cache) {
  free(cache->a[0]);
  free(cache->a[1]);
  free(cache->predictions);
  cache->a[0] = NULL;
  cache->a[1] = NULL;
  cache->predictions = NULL;
  cache->input = NULL;
}
void matrixInitHEuniform(Matrix *matrix) {
  double limit = sqrt(6.0 / matrix->cols);
  for (int i = 0; i < matrix->rows * matrix->cols; i++) {
    double u = (double)rand() / RAND_MAX;
    matrix->data[i] = (2.0 * u - 1) * limit;
  }
}
Network createNetwork(int h1, int h2) {
  Network network = {0};
  network.weights[0] = createMatrix(h1, 784);
  network.weights[1] = createMatrix(h2, h1);
  network.weights[2] = createMatrix(10, h2);
  for (int i = 0; i < 3; i++) {
    matrixInitHEuniform(&network.weights[i]);
  }
  network.biases[0] = calloc(h1, sizeof(double));
  network.biases[1] = calloc(h2, sizeof(double));
  network.biases[2] = calloc(10, sizeof(double));
  return network;
}
static double arrMax(double *arr, int arrLen) {
  assert(arrLen > 0);
  double mx = arr[0];
  for (int i = 1; i < arrLen; i++) {
    if (arr[i] > mx) {
      mx = arr[i];
    }
  }
  return mx;
}
double RElU(double x) {
  if (x < 0) {
    return 0;
  }
  return x;
}
void softMax(double *predictions) {
  double sm = 0;
  double mx = arrMax(predictions, 10);
  for (int i = 0; i < 10; i++) {
    sm += exp(predictions[i] - mx);
  }
  for (int i = 0; i < 10; i++) {
    predictions[i] = exp(predictions[i] - mx) / sm;
  }
}
double *forwardPass(Network *net, double *vector, ForwardCache *cache) {
  cache->input = vector;
  double *z1 = matrix_multiply_vector(&net->weights[0], vector, 784);
  if (z1 == NULL) {
    freeForwardCache(cache);
    return NULL;
  }
  for (int i = 0; i < net->weights[0].rows; i++) {
    z1[i] = RElU(z1[i] + net->biases[0][i]);
  }
  cache->a[0] = z1;
  double *z2 =
      matrix_multiply_vector(&net->weights[1], z1, net->weights[1].cols);
  if (z2 == NULL) {
    freeForwardCache(cache);
    return NULL;
  }
  for (int i = 0; i < net->weights[1].rows; i++) {
    z2[i] = RElU(z2[i] + net->biases[1][i]);
  }
  cache->a[1] = z2;
  double *predictions =
      matrix_multiply_vector(&net->weights[2], z2, net->weights[2].cols);
  if (predictions == NULL) {
    freeForwardCache(cache);
    return NULL;
  }
  for (int i = 0; i < 10; i++) {
    predictions[i] = predictions[i] + net->biases[2][i];
  }
  softMax(predictions);
  cache->predictions = predictions;
  return predictions;
}
