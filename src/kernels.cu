#include "../include/kernels.cuh"

/**
 * Version 3: GPU Basic (Naive Parallel)
 * Each thread computes PageRank for one page by scanning all pages
 */
__global__ void pageRankKernel_Naive(
    const int* outLinks,
    const int* offsets,
    const int* outDegree,
    const float* oldRank,
    float* newRank,
    int numPages,
    float dampingFactor)
{
    int pageId = blockIdx.x * blockDim.x + threadIdx.x;

    if (pageId < numPages) {
        float sum = 0.0f;

        // Sum contributions from all pages linking to this page
        for (int i = 0; i < numPages; i++) {
            int start = offsets[i];
            int end = offsets[i + 1];

            for (int j = start; j < end; j++) {
                if (outLinks[j] == pageId) {
                    sum += oldRank[i] / outDegree[i];
                    break;
                }
            }
        }

        newRank[pageId] = (1.0f - dampingFactor) / numPages + dampingFactor * sum;
    }
}

/**
 * Version 4: GPU Coalesced (Memory Optimization)
 * Uses transposed graph for coalesced memory access
 */
__global__ void pageRankKernel_Coalesced(
    const int* inLinks,
    const int* inOffsets,
    const int* outDegree,
    const float* oldRank,
    float* newRank,
    int numPages,
    float dampingFactor)
{
    int pageId = blockIdx.x * blockDim.x + threadIdx.x;

    if (pageId < numPages) {
        float sum = 0.0f;

        int start = inOffsets[pageId];
        int end = inOffsets[pageId + 1];

        for (int i = start; i < end; i++) {
            int srcPage = inLinks[i];
            sum += oldRank[srcPage] / outDegree[srcPage];
        }

        newRank[pageId] = (1.0f - dampingFactor) / numPages + dampingFactor * sum;
    }
}

/**
 * Version 5: GPU Shared Memory (Tiling)
 * Uses shared memory to cache frequently accessed rank data
 * Benefits graphs with clustering (many intra-block edges)
 */
__global__ void pageRankKernel_SharedMemory(
    const int* inLinks,
    const int* inOffsets,
    const int* outDegree,
    const float* oldRank,
    float* newRank,
    int numPages,
    float dampingFactor)
{
    // Shared memory for this block's rank and degree data
    __shared__ float s_ranks[256];
    __shared__ int s_degrees[256];
    
    int pageId = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load block's data into shared memory
    if (pageId < numPages) {
        s_ranks[tid] = oldRank[pageId];
        s_degrees[tid] = outDegree[pageId];
    }
    __syncthreads();
    
    if (pageId < numPages) {
        float sum = 0.0f;
        int start = inOffsets[pageId];
        int end = inOffsets[pageId + 1];
        
        // Compute block boundaries
        int blockStart = blockIdx.x * blockDim.x;
        int blockEnd = min(blockStart + blockDim.x, numPages);
        
        for (int i = start; i < end; i++) {
            int srcPage = inLinks[i];
            
            // Check if source page is in current block (shared memory)
            if (srcPage >= blockStart && srcPage < blockEnd) {
                // Fast path: use shared memory
                int localIdx = srcPage - blockStart;
                sum += s_ranks[localIdx] / s_degrees[localIdx];
            } else {
                // Slow path: use global memory
                sum += oldRank[srcPage] / outDegree[srcPage];
            }
        }
        
        newRank[pageId] = (1.0f - dampingFactor) / numPages + dampingFactor * sum;
    }
}

/**
 * Version 6: GPU Warp-Optimized (Warp Primitives)
 * Uses warp shuffle operations for high-degree nodes
 * Optimal for power-law graphs with skewed degree distribution
 */
__global__ void pageRankKernel_WarpOptimized(
    const int* inLinks,
    const int* inOffsets,
    const int* outDegree,
    const float* oldRank,
    float* newRank,
    int numPages,
    float dampingFactor)
{
    int pageId = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % 32;
    
    if (pageId < numPages) {
        int start = inOffsets[pageId];
        int end = inOffsets[pageId + 1];
        int numInLinks = end - start;
        
        float sum = 0.0f;
        
        // For high-degree nodes (>32 edges), use warp-level cooperation
        if (numInLinks > 32) {
            // Each lane processes every 32nd element
            for (int i = start + laneId; i < end; i += 32) {
                int srcPage = inLinks[i];
                sum += oldRank[srcPage] / outDegree[srcPage];
            }
            
            // Warp-level reduction using shuffle
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
            }
            
            // Only first lane writes the result
            if (laneId == 0) {
                newRank[pageId] = (1.0f - dampingFactor) / numPages + dampingFactor * sum;
            }
        } else {
            // Standard processing for low-degree nodes
            for (int i = start; i < end; i++) {
                int srcPage = inLinks[i];
                sum += oldRank[srcPage] / outDegree[srcPage];
            }
            newRank[pageId] = (1.0f - dampingFactor) / numPages + dampingFactor * sum;
        }
    }
}
