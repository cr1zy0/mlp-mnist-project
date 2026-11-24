#include "../include/network.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
double calculateCrossEntropyLoss(double *predictions, int label) {
  return -log(predictions[label]);
}

Gradients createGradients(const Network *network) {
  Gradients grad = {0};
  for (int i = 0; i < 3; i++) {
    grad.biases_gradients[i] =
        malloc(sizeof(double) * network->weights[i].rows);
    if (grad.biases_gradients[i] == NULL) {
      freeGradients(&grad);
      return grad;
    }
    grad.weights_gradients[i] =
        createMatrix(network->weights[i].rows, network->weights[i].cols);
    if (grad.weights_gradients[i].data == NULL) {
      freeGradients(&grad);
      return grad;
    }
    grad.deltas[i] = malloc(sizeof(double) * network->weights[i].rows);
    if (grad.deltas[i] == NULL) {
      freeGradients(&grad);
      return grad;
    }
  }
  return grad;
}
void freeGradients(Gradients *grad) {
  for (int i = 0; i < 3; i++) {
    free(grad->biases_gradients[i]);
    grad->biases_gradients[i] = NULL;
    freeMatrix(&grad->weights_gradients[i]);
    free(grad->deltas[i]);
    grad->deltas[i] = NULL;
  }
}

void zeroGradients(Gradients *gradients) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < gradients->weights_gradients[i].cols *
                            gradients->weights_gradients[i].rows;
         j++) {
      gradients->weights_gradients[i].data[j] = 0;
    }
  }
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < gradients->weights_gradients[i].rows; j++) {
      gradients->biases_gradients[i][j] = 0;
    }
  }
}
static void calculateOutputDelta(Gradients *grad, const ForwardCache *cache,
                                 int label) {
  for (int i = 0; i < grad->weights_gradients[2].rows; i++) {
    if (i == label) {
      grad->deltas[2][i] = cache->predictions[i] - 1;
    } else {
      grad->deltas[2][i] = cache->predictions[i];
    }
  }
}
void calculatePrevDelta(Gradients *grad, const ForwardCache *cache,
                        const Network *net, int layer) {
  if (layer < 0) {
    return;
  }
  for (int i = 0; i < net->weights[layer].rows; i++) {
    if (cache->a[layer][i] == 0) {
      grad->deltas[layer][i] = 0;
    } else {
      double sm = 0;
      for (int j = 0; j < net->weights[layer + 1].rows; j++) {
        sm += grad->deltas[layer + 1][j] *
              matrix_get(&net->weights[layer + 1], j, i);
      }
      grad->deltas[layer][i] = sm;
    }
  }
}
static void accumulateLayerGradients(Gradients *grad, const ForwardCache *cache,
                                     int layer) {
  for (int i = 0; i < grad->weights_gradients[layer].rows; i++) {
    for (int j = 0; j < grad->weights_gradients[layer].cols; j++) {
      if (layer == 0) {
        grad->weights_gradients->data[i * grad->weights_gradients->cols + j] +=
            grad->deltas[layer][i] * cache->input[j];
        continue;
      }
      grad->weights_gradients[layer]
          .data[i * grad->weights_gradients[layer].cols + j] +=
          grad->deltas[layer][i] * cache->a[layer - 1][j];
    }
    grad->biases_gradients[layer][i] += grad->deltas[layer][i];
  }
}
void backwardPass(Gradients *grad, const ForwardCache *cache,
                  const Network *net, int label) {
  calculateOutputDelta(grad, cache, label);
  for (int i = 2; i >= 0; i--) {
    accumulateLayerGradients(grad, cache, i);
    calculatePrevDelta(grad, cache, net, i - 1);
  }
}
void updateNetwork(Network *net, const Gradients *grad, double learningRate,
                   int batchSize) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < net->weights[i].cols * net->weights[i].rows; j++) {
      net->weights[i].data[j] =
          net->weights[i].data[j] -
          grad->weights_gradients[i].data[j] * learningRate / batchSize;
    }
    for (int j = 0; j < net->weights[i].rows; j++) {
      net->biases[i][j] = net->biases[i][j] - grad->biases_gradients[i][j] *
                                                  learningRate / batchSize;
    }
  }
}
int getPrediction(double *predictions) {
  double mx = predictions[0];
  int mxi = 0;
  for (int i = 1; i < 10; i++) {
    if (predictions[i] > mx) {
      mx = predictions[i];
      mxi = i;
    }
  }
  return mxi;
}

int saveWeights(const Network *network, const char *path) {
  FILE *file = fopen(path, "w");
  if (file == NULL)
    return 0;
  size_t layerCount = sizeof(network->weights) / sizeof(network->weights[0]);
  if (fprintf(file, "%d,%zu\n", 6767, layerCount) < 0) {
    fclose(file);
    return 0;
  }

  for (size_t i = 0; i < layerCount; i++) {
    const Matrix *weights = &network->weights[i];
    fprintf(file, "%d,%d\n", weights->rows, weights->cols);

    for (int j = 0; j < weights->rows * weights->cols - 1; j++)
      fprintf(file, "%.17g,", weights->data[j]);
    fprintf(file, "%.17g\n", weights->data[weights->rows * weights->cols - 1]);

    for (int j = 0; j < weights->rows - 1; j++)
      fprintf(file, "%.17g,", network->biases[i][j]);
    fprintf(file, "%.17g\n", network->biases[i][weights->rows - 1]);
  }
  int writeFailed = ferror(file);
  int closeResult = fclose(file);
  return !writeFailed && closeResult == 0;
}

int loadWeights(Network *network, const char *path) {
  if (network == NULL || path == NULL)
    return 0;

  size_t layerCount = sizeof(network->weights) / sizeof(network->weights[0]);
  size_t total = 0;
  size_t capacity = SIZE_MAX / sizeof(double);

  for (size_t i = 0; i < layerCount; i++) {
    const Matrix *weights = &network->weights[i];
    if (weights->rows <= 0 || weights->cols <= 0 || weights->data == NULL ||
        network->biases[i] == NULL)
      return 0;

    size_t rows = (size_t)weights->rows;
    size_t cols = (size_t)weights->cols;
    if (cols + 1 > (capacity - total) / rows)
      return 0;

    total += rows * (cols + 1);
  }

  FILE *file = fopen(path, "r");
  if (file == NULL)
    return 0;

  double *values = malloc(total * sizeof(*values));
  char line[128];
  char expected[128];
  int success = 0;

  if (values == NULL)
    goto cleanup;

  snprintf(expected, sizeof expected, "6767,%zu\n", layerCount);
  if (fgets(line, sizeof line, file) == NULL || strcmp(line, expected) != 0)
    goto cleanup;

  size_t offset = 0;
  for (size_t i = 0; i < layerCount; i++) {
    const Matrix *weights = &network->weights[i];
    snprintf(expected, sizeof expected, "%d,%d\n", weights->rows,
             weights->cols);
    if (fgets(line, sizeof line, file) == NULL || strcmp(line, expected) != 0)
      goto cleanup;

    size_t biasCount = (size_t)weights->rows;
    size_t weightCount = biasCount * (size_t)weights->cols;
    size_t valueCount = weightCount + biasCount;

    for (size_t j = 0; j < valueCount; j++) {
      int separator =
          (j + 1 == weightCount || j + 1 == valueCount) ? '\n' : ',';
      if (fscanf(file, "%lf", &values[offset + j]) != 1 ||
          fgetc(file) != separator)
        goto cleanup;
    }

    offset += valueCount;
  }

  if (fgetc(file) != EOF || ferror(file))
    goto cleanup;

  success = 1;

cleanup:
  if (fclose(file) != 0)
    success = 0;

  if (success) {
    size_t offset = 0;
    for (size_t i = 0; i < layerCount; i++) {
      Matrix *weights = &network->weights[i];
      size_t biasCount = (size_t)weights->rows;
      size_t weightCount = biasCount * (size_t)weights->cols;

      memcpy(weights->data, values + offset, weightCount * sizeof(double));
      offset += weightCount;
      memcpy(network->biases[i], values + offset, biasCount * sizeof(double));
      offset += biasCount;
    }
  }

  free(values);
  return success;
}
