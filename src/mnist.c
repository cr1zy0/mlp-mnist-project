#include "../include/mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>

static int read_4bytes(FILE *file, uint32_t *out) // функция для чтения 4 байт из файла по пути, юзается для чтения заголовков
{
    uint8_t buffer[4];
    if (fread(&buffer, 1, 4, file) < 4)
        return 0;
    uint32_t value = 0;
    for (int i = 0; i < 4; i++)
        value = (value << 8) | buffer[i];
    *out = value;
    return 1;
}

static int get_magic_n_count(FILE *file, uint32_t expected_magic, uint32_t *count) // функция для сверки магического числа и чтения количества объектов из файла
{                                                                                  // универсальна для images и labels
    uint32_t magic_number = 0;
    if (!(read_4bytes(file, &magic_number) == 1 && magic_number == expected_magic))
        return 0;

    if (read_4bytes(file, count) == 0)
        return 0;
    return 1;
}

Dataset *loadDataset(const char *imagesPath, const char *labelsPath)
{
    FILE *images_file = fopen(imagesPath, "rb");
    if (images_file == NULL) return NULL;
    FILE *labels_file = fopen(labelsPath, "rb");
    if (labels_file == NULL)
    {
        fclose(images_file);
        return NULL;
    }
    uint32_t images_count, labels_count;
    if (get_magic_n_count(images_file, 2051, &images_count) == 0)
    {
        fclose(images_file);
        fclose(labels_file);
        return NULL;
    }
    if (get_magic_n_count(labels_file, 2049, &labels_count) == 0)
    {
        fclose(images_file);
        fclose(labels_file);
        return NULL;
    }

    if (images_count != labels_count || images_count == 0 || images_count > INT_MAX)
    {
        fclose(images_file);
        fclose(labels_file);
        return NULL;
    }

    uint32_t width, height;
    if (read_4bytes(images_file, &height) == 0 || read_4bytes(images_file, &width) == 0 || width != height || width != 28)
    {
        fclose(images_file);
        fclose(labels_file);
        return NULL;
    }

    Dataset *dataset = malloc(sizeof(Dataset));
    double *pixels = malloc(sizeof(double) * images_count * 784);
    uint8_t *labels = malloc(sizeof(uint8_t) * labels_count);
    if (dataset == NULL || pixels == NULL || labels == NULL)
    {
        free(dataset);
        free(pixels);
        free(labels);
        fclose(images_file);
        fclose(labels_file);
        return NULL;
    }
    dataset->count = images_count;
    dataset->pixels = pixels;
    dataset->labels = labels;
    uint8_t buffer[width*height];
    int im_length = width*height;
    for (size_t i = 0; i < images_count; i++)
    {
        uint8_t curr_label;
        if (fread(&buffer, 1, im_length, images_file) != im_length || fread(&curr_label, 1, 1, labels_file) != 1 || curr_label > 9)
        {
            fclose(images_file);
            fclose(labels_file);
            freeDataset(dataset);
            return NULL;
        }
        for (size_t j = 0; j < im_length; j++)
            pixels[i * 784 + j] = buffer[j] / 255.0;
        labels[i] = curr_label;
    }
    fclose(images_file);
    fclose(labels_file);
    return dataset;
}

void freeDataset(Dataset *dataset)
{
    if (dataset == NULL)
        return;
    free(dataset->pixels);
    free(dataset->labels);
    free(dataset);
}

Dataset *createDataset(const int count)
{
    Dataset *dataset = malloc(sizeof(Dataset));
    dataset->count = count;
    dataset->pixels = malloc(sizeof(double) * 784 * count);
    dataset->labels = malloc(sizeof(double) * count);
    return dataset;
}