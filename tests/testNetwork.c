#include "../include/network.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double sampleLoss(Network *net, double *input, int label) {
  ForwardCache cache = {0};
  double *p = forwardPass(net, input, &cache);
  assert(p != NULL);
  double loss = calculateCrossEntropyLoss(p, label);
  freeForwardCache(&cache);
  return loss;
}

static double batchLoss(Network *net, double *first, double *second) {
  return (sampleLoss(net, first, 1) + sampleLoss(net, second, 2)) / 2.0;
}

static double numericalGradient(Network *net, double *parameter,
                                double *first, double *second) {
  double original = *parameter;
  double eps = 1e-4;
  *parameter = original + eps;
  double plus = batchLoss(net, first, second);
  *parameter = original - eps;
  double minus = batchLoss(net, first, second);
  *parameter = original;
  return (plus - minus) / (2.0 * eps);
}

static void testBackward(Network *net, double *input) {
  matrix_set(&net->weights[2], 1, 2, -0.5);
  Gradients grad = createGradients(net);
  double second[784] = {0.5};
  input[783] = 0.25;
  second[783] = 0.75;
  for (int l = 0; l < 3; l++) {
    assert(grad.weights_gradients[l].data != NULL);
    assert(grad.biases_gradients[l] != NULL && grad.deltas[l] != NULL);
  }
  zeroGradients(&grad);
  ForwardCache cache = {0};
  assert(forwardPass(net, input, &cache) != NULL);
  backwardPass(&grad, &cache, net, 1);
  freeForwardCache(&cache);
  assert(forwardPass(net, second, &cache) != NULL);
  backwardPass(&grad, &cache, net, 2);
  freeForwardCache(&cache);
  assert(fabs(grad.weights_gradients[0].data[0]) > 1e-6);

  printf("\nGradient check (two different samples):\n");
  for (int l = 0; l < 3; l++) {
    Matrix *w = &net->weights[l];
    Matrix *g = &grad.weights_gradients[l];
    double maxWeightError = 0.0, maxBiasError = 0.0;
    for (int i = 0; i < w->rows * w->cols; i++) {
      double numerical = numericalGradient(net, &w->data[i], input, second);
      double analytical = g->data[i] / 2.0;
      double error = fabs(numerical - analytical);
      assert(isfinite(error) && error < 1e-6);
      maxWeightError = fmax(maxWeightError, error);
    }
    for (int i = 0; i < w->rows; i++) {
      double numerical = numericalGradient(net, &net->biases[l][i],
                                           input, second);
      double analytical = grad.biases_gradients[l][i] / 2.0;
      double error = fabs(numerical - analytical);
      assert(isfinite(error) && error < 1e-6);
      maxBiasError = fmax(maxBiasError, error);
    }
    printf("Layer %d: %d weights, %d biases, max errors %.3e / %.3e [OK]\n",
           l + 1, w->rows * w->cols, w->rows, maxWeightError, maxBiasError);
  }

  double before = batchLoss(net, input, second);
  double oldWeight = net->weights[2].data[3];
  double oldBias = net->biases[2][1];
  updateNetwork(net, &grad, 0.01, 2);
  assert(fabs(net->weights[2].data[3] -
              (oldWeight - 0.005 * grad.weights_gradients[2].data[3])) < 1e-12);
  assert(fabs(net->biases[2][1] -
              (oldBias - 0.005 * grad.biases_gradients[2][1])) < 1e-12);
  double after = batchLoss(net, input, second);
  printf("Batch loss: %.8f -> %.8f\n", before, after);
  assert(after < before);

  zeroGradients(&grad);
  for (int l = 0; l < 3; l++) {
    Matrix *g = &grad.weights_gradients[l];
    for (int i = 0; i < g->rows * g->cols; i++)
      assert(g->data[i] == 0.0);
    for (int i = 0; i < g->rows; i++)
      assert(grad.biases_gradients[l][i] == 0.0);
  }
  printf("Gradient reset: [OK]\n");
  freeGradients(&grad);
  for (int l = 0; l < 3; l++) {
    assert(grad.weights_gradients[l].data == NULL);
    assert(grad.biases_gradients[l] == NULL && grad.deltas[l] == NULL);
  }
  printf("Gradient cleanup: [OK]\n");
}

int main(void) {
  srand(42);
  Network net = createNetwork(2, 3);
  int rows[] = {2, 3, 10};
  int cols[] = {784, 2, 3};

  printf("Network initialization:\n");
  for (int l = 0; l < 3; l++) {
    Matrix *w = &net.weights[l];
    assert(w->data != NULL && net.biases[l] != NULL);
    assert(w->rows == rows[l] && w->cols == cols[l]);
    double limit = sqrt(6.0 / w->cols);
    for (int i = 0; i < w->rows * w->cols; i++) {
      assert(isfinite(w->data[i]));
      assert(fabs(w->data[i]) <= limit);
    }
    for (int i = 0; i < w->rows; i++) {
      assert(net.biases[l][i] == 0.0);
    }
    printf("Layer %d: %d x %d, He limit = %.6f, biases = 0 [OK]\n",
           l + 1, w->rows, w->cols, limit);
  }

  for (int l = 0; l < 3; l++) {
    Matrix *w = &net.weights[l];
    for (int i = 0; i < w->rows * w->cols; i++) {
      w->data[i] = 0.0;
    }
  }

  double input[784] = {1.0};
  matrix_set(&net.weights[0], 0, 0, 1.0);
  matrix_set(&net.weights[0], 1, 0, -2.0);
  net.biases[0][0] = 1.0;

  matrix_set(&net.weights[1], 0, 0, 1.0);
  matrix_set(&net.weights[1], 1, 0, -1.0);
  matrix_set(&net.weights[1], 2, 0, 2.0);
  net.biases[1][0] = 1.0;
  net.biases[1][2] = 1.0;

  matrix_set(&net.weights[2], 1, 0, 2.0);
  matrix_set(&net.weights[2], 1, 2, -1.0);
  for (int i = 0; i < 10; i++) {
    net.biases[2][i] = 1000.0;
  }

  ForwardCache cache = {0};
  double *predictions = forwardPass(&net, input, &cache);
  assert(predictions != NULL);
  assert(cache.input == input);
  assert(cache.predictions == predictions);

  printf("\nForward pass with known weights:\n");
  printf("a1: %.1f %.1f (expected: 2 0)\n", cache.a[0][0],
         cache.a[0][1]);
  printf("a2: %.1f %.1f %.1f (expected: 3 0 5)\n", cache.a[1][0],
         cache.a[1][1], cache.a[1][2]);
  assert(cache.a[0][0] == 2.0 && cache.a[0][1] == 0.0);
  assert(cache.a[1][0] == 3.0 && cache.a[1][1] == 0.0 &&
         cache.a[1][2] == 5.0);

  printf("\nSoftmax with large logits (1000, 1001, 1000, ...):\n");
  double sum = 0.0;
  for (int i = 0; i < 10; i++) {
    double expected = (i == 1 ? exp(1.0) : 1.0) / (9.0 + exp(1.0));
    assert(isfinite(predictions[i]));
    assert(fabs(predictions[i] - expected) < 1e-12);
    printf("Class %d: %.8f (expected %.8f)\n", i, predictions[i],
           expected);
    sum += predictions[i];
  }
  printf("Probability sum: %.12f\n", sum);
  assert(fabs(sum - 1.0) < 1e-12);

  freeForwardCache(&cache);
  assert(cache.input == NULL && cache.a[0] == NULL &&
         cache.a[1] == NULL && cache.predictions == NULL);
  freeForwardCache(&cache);
  predictions = forwardPass(&net, input, &cache);
  assert(predictions != NULL);
  assert(fabs(predictions[1] - exp(1.0) / (9.0 + exp(1.0))) < 1e-12);
  freeForwardCache(&cache);
  printf("\nCache cleanup and reuse: [OK]\n");

  testBackward(&net, input);

  for (int l = 0; l < 3; l++) {
    freeMatrix(&net.weights[l]);
    free(net.biases[l]);
  }
  printf("\nAll network tests passed!\n");
  return 0;
}
