#ifndef DEFINE_DATASET
    #define DEFINE_DATASET
    
    typedef struct {
        int count;
        double *pixels;
        unsigned char *labels;
    }Dataset;

    Dataset *loadDataset(const char *imagesPath, const char *labelsPath);
    void freeDataset(Dataset *dataset);
#endif