#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define DAMPING_FACTOR 0.85f
#define EPSILON 1e-6f
#define MAX_ITERATIONS 100

// Error checking macro
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
 * CUDA kernel for PageRank computation
 */
__global__ void pageRankKernel(
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
 * Optimized CUDA kernel using transposed graph
 */
__global__ void pageRankKernelOptimized(
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
 * CPU implementation for verification
 */
void pageRankCPU(
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

        for (int i = 0; i < numPages; i++) {
            ranks[i] = tempRanks[i];
        }
    }

    free(tempRanks);
}

/**
 * Transpose graph for optimized version
 */
void transposeGraph(
    const int* outLinks,
    const int* outOffsets,
    int numPages,
    int numEdges,
    int** inLinks,
    int** inOffsets)
{
    *inLinks = (int*)malloc(numEdges * sizeof(int));
    *inOffsets = (int*)malloc((numPages + 1) * sizeof(int));

    int* inDegree = (int*)calloc(numPages, sizeof(int));
    for (int i = 0; i < numPages; i++) {
        for (int j = outOffsets[i]; j < outOffsets[i + 1]; j++) {
            inDegree[outLinks[j]]++;
        }
    }

    (*inOffsets)[0] = 0;
    for (int i = 0; i < numPages; i++) {
        (*inOffsets)[i + 1] = (*inOffsets)[i] + inDegree[i];
    }

    int* currentPos = (int*)calloc(numPages, sizeof(int));
    for (int i = 0; i < numPages; i++) {
        for (int j = outOffsets[i]; j < outOffsets[i + 1]; j++) {
            int dest = outLinks[j];
            int pos = (*inOffsets)[dest] + currentPos[dest];
            (*inLinks)[pos] = i;
            currentPos[dest]++;
        }
    }

    free(inDegree);
    free(currentPos);
}

/**
 * Create sample graph
 */
void createSampleGraph(
    int** outLinks,
    int** offsets,
    int** outDegree,
    int* numPages,
    int* numEdges)
{
    *numPages = 7;
    *numEdges = 12;

    *outLinks = (int*)malloc(*numEdges * sizeof(int));
    *offsets = (int*)malloc((*numPages + 1) * sizeof(int));
    *outDegree = (int*)malloc(*numPages * sizeof(int));

    int links[] = {1, 2, 2, 0, 2, 4, 3, 5, 4, 5, 4, 0};
    for (int i = 0; i < *numEdges; i++) {
        (*outLinks)[i] = links[i];
    }

    int offs[] = {0, 2, 3, 4, 6, 8, 9, 12};
    for (int i = 0; i <= *numPages; i++) {
        (*offsets)[i] = offs[i];
    }

    for (int i = 0; i < *numPages; i++) {
        (*outDegree)[i] = (*offsets)[i + 1] - (*offsets)[i];
    }
}

int main() {
    int *h_outLinks, *h_offsets, *h_outDegree;
    int numPages, numEdges;

    createSampleGraph(&h_outLinks, &h_offsets, &h_outDegree, &numPages, &numEdges);

    printf("\n" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("   PageRank Algorithm - CUDA Implementation\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n\n");
    printf("Number of pages: %d\n", numPages);
    printf("Number of edges: %d\n\n", numEdges);

    float* h_ranks = (float*)malloc(numPages * sizeof(float));
    for (int i = 0; i < numPages; i++) {
        h_ranks[i] = 1.0f / numPages;
    }

    printf("🖥️  Running CPU version...\n");
    float* h_ranksCPU = (float*)malloc(numPages * sizeof(float));
    for (int i = 0; i < numPages; i++) {
        h_ranksCPU[i] = 1.0f / numPages;
    }

    pageRankCPU(h_outLinks, h_offsets, h_outDegree, h_ranksCPU,
                numPages, DAMPING_FACTOR, MAX_ITERATIONS);

    printf("\nCPU PageRank Results:\n");
    for (int i = 0; i < numPages; i++) {
        printf("  Page %d: %.6f\n", i, h_ranksCPU[i]);
    }
    printf("\n");

    printf("🚀 Running GPU version (optimized)...\n");

    int *h_inLinks, *h_inOffsets;
    transposeGraph(h_outLinks, h_offsets, numPages, numEdges, &h_inLinks, &h_inOffsets);

    int *d_inLinks, *d_inOffsets, *d_outDegree;
    float *d_ranks, *d_newRanks;

    CUDA_CHECK(cudaMalloc(&d_inLinks, numEdges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_inOffsets, (numPages + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_outDegree, numPages * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ranks, numPages * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_newRanks, numPages * sizeof(float)));

    for (int i = 0; i < numPages; i++) {
        h_ranks[i] = 1.0f / numPages;
    }

    CUDA_CHECK(cudaMemcpy(d_inLinks, h_inLinks, numEdges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inOffsets, h_inOffsets, (numPages + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_outDegree, h_outDegree, numPages * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ranks, h_ranks, numPages * sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (numPages + threadsPerBlock - 1) / threadsPerBlock;

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        pageRankKernelOptimized<<<blocksPerGrid, threadsPerBlock>>>(
            d_inLinks, d_inOffsets, d_outDegree, d_ranks, d_newRanks,
            numPages, DAMPING_FACTOR);
        CUDA_CHECK(cudaGetLastError());

        float* temp = d_ranks;
        d_ranks = d_newRanks;
        d_newRanks = temp;
    }

    CUDA_CHECK(cudaMemcpy(h_ranks, d_ranks, numPages * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\nGPU PageRank Results:\n");
    for (int i = 0; i < numPages; i++) {
        printf("  Page %d: %.6f\n", i, h_ranks[i]);
    }
    printf("\n");

    printf("✅ Verification:\n");
    float maxDiff = 0.0f;
    for (int i = 0; i < numPages; i++) {
        float diff = fabs(h_ranks[i] - h_ranksCPU[i]);
        if (diff > maxDiff) maxDiff = diff;
    }
    printf("  Maximum difference: %e\n", maxDiff);
    if (maxDiff < 1e-4) {
        printf("  ✓ Results match!\n\n");
    } else {
        printf("  ✗ Results differ!\n\n");
    }

    free(h_outLinks);
    free(h_offsets);
    free(h_outDegree);
    free(h_ranks);
    free(h_ranksCPU);
    free(h_inLinks);
    free(h_inOffsets);

    cudaFree(d_inLinks);
    cudaFree(d_inOffsets);
    cudaFree(d_outDegree);
    cudaFree(d_ranks);
    cudaFree(d_newRanks);

    return 0;
}
