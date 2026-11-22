#include "matrix.h"

typedef struct {
  Matrix weights[3];
  double *biases[3];
} Network;

typedef struct {
  const double *input;
  double *a[2];
  double *predictions;
} ForwardCache;

void freeForwardCache(ForwardCache *cache);
Network createNetwork(int h1, int h2);
double *forwardPass(Network *net, double *vector, ForwardCache *cache);
