#include "mnist.h"
#include "network.h"
void shuffleIndices(int *indices, int count);
void trainValidationSplit(const Dataset *ds, Dataset *train,
                          Dataset *validation);
void networkFit(Network *net, const Dataset *train, const Dataset *validation,
                double learningRate, int numEpochs, int batchSize);
void printNetworkMetrics(int curent_ep, int ammountEp, double train_loss,
                         double val_loss, double train_accuracy,
                         double val_accuracy, double elapsed_time);
void networkEvaluate(Network *net, const Dataset *test, double *val_accuracy,
                     double *val_loss);
