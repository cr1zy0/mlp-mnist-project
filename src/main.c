#include "..//include//mnist.h"
#include "..//include//network.h"
#include "..//include//train.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
  Network net = createNetwork(128, 64);
  if (argc == 4 && strcmp(argv[1], "--eval") == 0) {
    if (!loadWeights(&net, "weights.csv")) {
      fprintf(stderr, "Failed to load weights\n");
      return 1;
    }
    Dataset *eval = loadDataset(argv[2], argv[3]);
    if (eval == NULL) {
      printf("failed to read DS");
      return 1;
    }
    double evalLoss = 0;
    double evalAcc = 0;
    networkEvaluate(&net, eval, &evalAcc, &evalLoss);
    evalLoss /= eval->count;
    evalAcc /= eval->count;
    printf("Test | Loss: %.4f | Accuracy: %.2f%%\n", evalLoss, evalAcc * 100);
    freeDataset(eval);
  } else if (argc == 4 && strcmp(argv[1], "--inference") == 0) {
    if (!loadWeights(&net, "weights.csv")) {
      fprintf(stderr, "Failed to load weights\n");
      return 1;
    }
    networkInference(&net, argv[2], argv[3]);
  } else if (argc == 2 && strcmp(argv[1], "--train") == 0) {

    Dataset *ds = loadDataset("data/train-images.idx3-ubyte",
                              "data/train-labels.idx1-ubyte");
    if (ds == NULL) {
      printf("failed to load DS");
      return 1;
    }
    Dataset *train = createDataset(50000);
    Dataset *test = createDataset(10000);
    trainValidationSplit(ds, train, test);
    freeDataset(ds);
    networkFit(&net, train, test, 0.01, 10, 16);
    saveWeights(&net, "weights.csv");
    freeDataset(test);
    freeDataset(train);
  } else {
    printf(
        "Неправильный запуск, достпуные флаги: --train, --inference, --eval");
    return 1;
  }
}
