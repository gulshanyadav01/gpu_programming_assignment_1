#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>

/**
 * CUDA error checking macro
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * Timing utilities
 */
typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
} GpuTimer;

void startGpuTimer(GpuTimer* timer);
float stopGpuTimer(GpuTimer* timer);

/**
 * CPU timing
 */
typedef struct {
    double start_time;
} CpuTimer;

void startCpuTimer(CpuTimer* timer);
double stopCpuTimer(CpuTimer* timer);

/**
 * Verification utilities
 */
float verifyResults(const float* results1, const float* results2, int numPages);
void printRanks(const float* ranks, int numPages, const char* label);

/**
 * CPU PageRank implementations
 */

// Version 1: CPU Naive
void pageRankCPU_Naive(
    const int* outLinks,
    const int* offsets,
    const int* outDegree,
    float* ranks,
    int numPages,
    float dampingFactor,
    int maxIterations
);

// Version 2: CPU Optimized (Transposed Graph)
void pageRankCPU_Optimized(
    const int* inLinks,
    const int* inOffsets,
    const int* outDegree,
    float* ranks,
    int numPages,
    float dampingFactor,
    int maxIterations
);

#endif // UTILS_H
