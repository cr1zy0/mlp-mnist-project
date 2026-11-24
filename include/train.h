#include "mnist.h"
#include "network.h"
void shuffleIndices(int *indices, int count);
void trainValidationSplit(const Dataset *ds, Dataset *train,
                          Dataset *validation);
void networkFit(Network *net, const Dataset *train, const Dataset *validation,
                double learningRate, int numEpochs, int batchSize);
void printAccuracy(int ep, double train_accuracy, double val_accuracy);
double networkEvaluate(Network *net, const Dataset *test);
