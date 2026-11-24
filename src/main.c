#include "..//include//mnist.h"
#include "..//include//network.h"
#include "..//include//train.h"

int main() {
  Dataset *ds = loadDataset("", "");
  Dataset *eval = loadDataset("", "");
  Dataset *train = createDataset(50000);
  Dataset *test = createDataset(10000);
  trainValidationSplit(ds, train, test);
  freeDataset(ds);
  Network net = createNetwork(128, 64);
  networkFit(&net, train, test, 0.01, 10, 16);
  networkEvaluate(&net, eval);
  freeDataset(test);
  freeDataset(train);
  freeDataset(eval);
}
