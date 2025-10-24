#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

/**
 * CUDA kernel declarations for PageRank implementations
 */

// Version 3: GPU Basic (Naive Parallel)
__global__ void pageRankKernel_Naive(
    const int* outLinks,
    const int* offsets,
    const int* outDegree,
    const float* oldRank,
    float* newRank,
    int numPages,
    float dampingFactor
);

// Version 4: GPU Coalesced (Memory Optimization)
__global__ void pageRankKernel_Coalesced(
    const int* inLinks,
    const int* inOffsets,
    const int* outDegree,
    const float* oldRank,
    float* newRank,
    int numPages,
    float dampingFactor
);

// Version 5: GPU Shared Memory (Tiling)
__global__ void pageRankKernel_SharedMemory(
    const int* inLinks,
    const int* inOffsets,
    const int* outDegree,
    const float* oldRank,
    float* newRank,
    int numPages,
    float dampingFactor
);

// Version 6: GPU Warp-Optimized (Warp Primitives)
__global__ void pageRankKernel_WarpOptimized(
    const int* inLinks,
    const int* inOffsets,
    const int* outDegree,
    const float* oldRank,
    float* newRank,
    int numPages,
    float dampingFactor
);

#endif // KERNELS_CUH
