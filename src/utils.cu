#include "../include/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

// GPU Timer functions
void startGpuTimer(GpuTimer* timer) {
    cudaEventCreate(&timer->start);
    cudaEventCreate(&timer->stop);
    cudaEventRecord(timer->start, 0);
}

float stopGpuTimer(GpuTimer* timer) {
    float elapsedTime;
    cudaEventRecord(timer->stop, 0);
    cudaEventSynchronize(timer->stop);
    cudaEventElapsedTime(&elapsedTime, timer->start, timer->stop);
    cudaEventDestroy(timer->start);
    cudaEventDestroy(timer->stop);
    return elapsedTime;
}

// CPU Timer functions
void startCpuTimer(CpuTimer* timer) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    timer->start_time = tv.tv_sec + tv.tv_usec / 1000000.0;
}

double stopCpuTimer(CpuTimer* timer) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double end_time = tv.tv_sec + tv.tv_usec / 1000000.0;
    return (end_time - timer->start_time) * 1000.0; // Return in milliseconds
}

// Verification function
float verifyResults(const float* results1, const float* results2, int numPages) {
    float maxDiff = 0.0f;
    for (int i = 0; i < numPages; i++) {
        float diff = fabs(results1[i] - results2[i]);
        if (diff > maxDiff) {
            maxDiff = diff;
        }
    }
    return maxDiff;
}

void printRanks(const float* ranks, int numPages, const char* label) {
    printf("\n%s:\n", label);
    int printCount = numPages < 10 ? numPages : 10;
    for (int i = 0; i < printCount; i++) {
        printf("  Page %d: %.6f\n", i, ranks[i]);
    }
    if (numPages > 10) {
        printf("  ... (%d more pages)\n", numPages - 10);
    }
}

// Version 1: CPU Naive Implementation
void pageRankCPU_Naive(
    const int* outLinks,
    const int* offsets,
    const int* outDegree,
    float* ranks,
    int numPages,
    float dampingFactor,
    int maxIterations)
{
    float* tempRanks = (float*)malloc(numPages * sizeof(float));

    for (int iter = 0; iter < maxIterations; iter++) {
        for (int i = 0; i < numPages; i++) {
            float sum = 0.0f;

            // For each page, scan ALL other pages to find incoming links
            for (int j = 0; j < numPages; j++) {
                int start = offsets[j];
                int end = offsets[j + 1];

                for (int k = start; k < end; k++) {
                    if (outLinks[k] == i) {
                        sum += ranks[j] / outDegree[j];
                        break;
                    }
                }
            }

            tempRanks[i] = (1.0f - dampingFactor) / numPages + dampingFactor * sum;
        }

        // Copy results back
        for (int i = 0; i < numPages; i++) {
            ranks[i] = tempRanks[i];
        }
    }

    free(tempRanks);
}

// Version 2: CPU Optimized (using transposed graph)
void pageRankCPU_Optimized(
    const int* inLinks,
    const int* inOffsets,
    const int* outDegree,
    float* ranks,
    int numPages,
    float dampingFactor,
    int maxIterations)
{
    float* tempRanks = (float*)malloc(numPages * sizeof(float));

    for (int iter = 0; iter < maxIterations; iter++) {
        for (int i = 0; i < numPages; i++) {
            float sum = 0.0f;
            int start = inOffsets[i];
            int end = inOffsets[i + 1];

            // Only iterate through actual incoming links (optimized)
            for (int j = start; j < end; j++) {
                int srcPage = inLinks[j];
                sum += ranks[srcPage] / outDegree[srcPage];
            }

            tempRanks[i] = (1.0f - dampingFactor) / numPages + dampingFactor * sum;
        }

        // Copy results back
        for (int i = 0; i < numPages; i++) {
            ranks[i] = tempRanks[i];
        }
    }

    free(tempRanks);
}
