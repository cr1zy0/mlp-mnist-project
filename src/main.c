#include "..//include//mnist.h"
#include "..//include//network.h"
#include "..//include//train.h"
#include <stdio.h>

int main() {
  Dataset *ds = loadDataset("data/train-images.idx3-ubyte",
                            "data/train-labels.idx1-ubyte");
  Dataset *eval =
      loadDataset("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
  Dataset *train = createDataset(50000);
  Dataset *test = createDataset(10000);
  trainValidationSplit(ds, train, test);
  freeDataset(ds);
  Network net = createNetwork(128, 64);
  networkFit(&net, train, test, 0.01, 10, 16);
  double evalLoss = 0;
  double evalAcc = 0;
  networkEvaluate(&net, eval, &evalAcc, &evalLoss);
  evalLoss /= eval->count;
  evalAcc /= eval->count;
  printf("Test | Loss: %.4f | Accuracy: %.2f%%\n", evalLoss, evalAcc * 100);
  freeDataset(test);
  freeDataset(train);
  freeDataset(eval);
}
