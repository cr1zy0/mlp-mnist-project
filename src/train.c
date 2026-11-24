#include "..//include/train.h"
#include "..//include/mnist.h"
#include "..//include/network.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
void shuffleIndices(int *indices, int count) {
  for (int i = count - 1; i >= 0; i--) {
    int j = rand() % (i + 1);
    int tmp = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp;
  }
}
void trainValidationSplit(const Dataset *ds, Dataset *train,
                          Dataset *validation) {
  int indices[ds->count];
  for (int i = 0; i < ds->count; i++) {
    indices[i] = i;
  }
  shuffleIndices(indices, ds->count);
  for (int i = 0; i < train->count; i++) {
    train->labels[i] = ds->labels[indices[i]];
    memcpy(train->pixels + i * 784, ds->pixels + indices[i] * 784,
           784 * sizeof(double));
  }
  for (int i = 0; i < validation->count; i++) {
    validation->labels[i] = ds->labels[indices[train->count + i]];
    memcpy(validation->pixels + i * 784,
           ds->pixels + indices[train->count + i] * 784, 784 * sizeof(double));
  }
}
void printNetworkMetrics(int current_ep, int amountEp, double train_loss,
                         double val_loss, double train_accuracy,
                         double val_accuracy, double elapsed_time) {
  printf("Epoch %02d/%02d | Loss: %.4f | Acc: %.2f%% | Val loss: %.4f | Val "
         "acc: %.2f%% | Time: %.2fs\n",
         current_ep + 1, amountEp, train_loss, train_accuracy * 100, val_loss,
         val_accuracy * 100, elapsed_time);
}
void networkEvaluate(Network *net, const Dataset *test, double *val_accuracy,
                     double *val_loss) {
  ForwardCache cache = {0};
  for (int i = 0; i < test->count; i++) {
    forwardPass(net, test->pixels + i * 784, &cache);
    *(val_loss) +=
        calculateCrossEntropyLoss(cache.predictions, test->labels[i]);
    if (getPrediction(cache.predictions) == test->labels[i]) {
      (*val_accuracy)++;
    }
    freeForwardCache(&cache);
  }
}
void networkFit(Network *net, const Dataset *train, const Dataset *validation,
                double learningRate, int numEpochs, int batchSize) {
  int indices[train->count];
  Gradients grad = createGradients(net);
  ForwardCache cache = {0};
  for (int i = 0; i < train->count; i++) {
    indices[i] = i;
  }
  for (int ep = 0; ep < numEpochs; ep++) {
    clock_t start = clock();
    shuffleIndices(indices, train->count);
    zeroGradients(&grad);
    double train_accuracy = 0;
    double val_accuracy = 0;
    double train_loss = 0;
    double val_loss = 0;

    for (int i = 0; i < train->count; i++) {
      forwardPass(net, train->pixels + indices[i] * 784, &cache);
      int prediction = getPrediction(cache.predictions);
      if (prediction == train->labels[indices[i]]) {
        train_accuracy++;
      }
      train_loss += calculateCrossEntropyLoss(cache.predictions,
                                              train->labels[indices[i]]);
      backwardPass(&grad, &cache, net, train->labels[indices[i]]);
      freeForwardCache(&cache);
      if ((i + 1) % batchSize == 0) {
        updateNetwork(net, &grad, learningRate, batchSize);
        zeroGradients(&grad);
      }
    }
    train_accuracy = train_accuracy / train->count;
    train_loss = train_loss / train->count;
    networkEvaluate(net, validation, &val_accuracy, &val_loss);
    val_accuracy = val_accuracy / validation->count;
    val_loss = val_loss / validation->count;
    double elapsedTime = (double)(clock() - start) / CLOCKS_PER_SEC;
    printNetworkMetrics(ep, numEpochs, train_loss, val_loss, train_accuracy,
                        val_accuracy, elapsedTime);
  }
  freeGradients(&grad);
}
