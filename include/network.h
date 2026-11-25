#ifndef NETWORK_H
  #define NETWORK_H

  #define MAGIC_NUMBER 6767

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
  typedef struct {
    Matrix weights_gradients[3];
    double *biases_gradients[3];
    double *deltas[3];
  } Gradients;

  void freeForwardCache(ForwardCache *cache);
  Network createNetwork(int h1, int h2);
  double *forwardPass(Network *net, double *vector, ForwardCache *cache);
  double calculateCrossEntropyLoss(double *predictions, int label);
  Gradients createGradients(const Network *network);
  void zeroGradients(Gradients *gradients);
  void freeGradients(Gradients *grad);
  void backwardPass(Gradients *grad, const ForwardCache *cache,
                    const Network *net, int label);
  void updateNetwork(Network *net, const Gradients *grad, double learningRate,
                    int batchSize);
  int getPrediction(double *predictions);
  void networkInference(Network *net, char *image_path, char *label_path);

  int saveWeights(const Network *network, const char *path);
  int loadWeights(Network *network, const char *path);

#endif
