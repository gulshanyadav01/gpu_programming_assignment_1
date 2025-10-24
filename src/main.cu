#include "../include/graph.h"
#include "../include/kernels.cuh"
#include "../include/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DAMPING_FACTOR 0.85f
#define MAX_ITERATIONS 100

/**
 * Run GPU PageRank with specified kernel
 */
float runGPU_PageRank(
    const char* version_name,
    void (*kernel)(const int*, const int*, const int*, const float*, float*, int, float),
    const Graph* graph,
    float* h_ranks,
    bool useTransposed)
{
    int numPages = graph->numPages;
    int numEdges = graph->numEdges;
    
    // Device memory
    int *d_links, *d_offsets, *d_outDegree;
    float *d_ranks, *d_newRanks;
    
    CUDA_CHECK(cudaMalloc(&d_links, numEdges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offsets, (numPages + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_outDegree, numPages * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ranks, numPages * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_newRanks, numPages * sizeof(float)));
    
    // Initialize ranks
    for (int i = 0; i < numPages; i++) {
        h_ranks[i] = 1.0f / numPages;
    }
    
    // Copy data to device
    if (useTransposed) {
        CUDA_CHECK(cudaMemcpy(d_links, graph->inLinks, numEdges * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_offsets, graph->inOffsets, (numPages + 1) * sizeof(int), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_links, graph->outLinks, numEdges * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_offsets, graph->offsets, (numPages + 1) * sizeof(int), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_outDegree, graph->outDegree, numPages * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ranks, h_ranks, numPages * sizeof(float), cudaMemcpyHostToDevice));
    
    // Configure kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numPages + threadsPerBlock - 1) / threadsPerBlock;
    
    // Start timing
    GpuTimer timer;
    startGpuTimer(&timer);
    
    // Run iterations
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_links, d_offsets, d_outDegree, d_ranks, d_newRanks,
            numPages, DAMPING_FACTOR);
        CUDA_CHECK(cudaGetLastError());
        
        // Swap buffers
        float* temp = d_ranks;
        d_ranks = d_newRanks;
        d_newRanks = temp;
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    float elapsed = stopGpuTimer(&timer);
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_ranks, d_ranks, numPages * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Cleanup
    cudaFree(d_links);
    cudaFree(d_offsets);
    cudaFree(d_outDegree);
    cudaFree(d_ranks);
    cudaFree(d_newRanks);
    
    printf("  %-25s %10.3f ms\n", version_name, elapsed);
    return elapsed;
}

/**
 * Main benchmarking program
 */
int main(int argc, char** argv) {
    printf("\n");
    printf("================================================================================\n");
    printf("                PageRank Algorithm - Complete CUDA Suite\n");
    printf("================================================================================\n\n");
    
    // Determine graph size
    int graphSize = 7; // Default: small sample graph
    bool useCustomSize = false;
    
    if (argc > 1) {
        graphSize = atoi(argv[1]);
        useCustomSize = true;
    }
    
    // Create graph
    Graph graph;
    if (useCustomSize && graphSize > 7) {
        printf("Generating random graph...\n");
        generateRandomGraph(&graph, graphSize, 5); // Average degree = 5
    } else {
        printf("Using sample graph (7 nodes)...\n");
        createSampleGraph(&graph);
        graphSize = 7;
    }
    
    printGraphStats(&graph);
    printf("\n");
    
    // Allocate results arrays
    float* cpuNaive = (float*)malloc(graph.numPages * sizeof(float));
    float* cpuOpt = (float*)malloc(graph.numPages * sizeof(float));
    float* gpuNaive = (float*)malloc(graph.numPages * sizeof(float));
    float* gpuCoalesced = (float*)malloc(graph.numPages * sizeof(float));
    float* gpuShared = (float*)malloc(graph.numPages * sizeof(float));
    float* gpuWarp = (float*)malloc(graph.numPages * sizeof(float));
    
    // Initialize all with same starting values
    for (int i = 0; i < graph.numPages; i++) {
        cpuNaive[i] = 1.0f / graph.numPages;
        cpuOpt[i] = 1.0f / graph.numPages;
    }
    
    printf("================================================================================\n");
    printf("                          BENCHMARKING RESULTS\n");
    printf("================================================================================\n\n");
    printf("Configuration:\n");
    printf("  Iterations: %d\n", MAX_ITERATIONS);
    printf("  Damping factor: %.2f\n", DAMPING_FACTOR);
    printf("  Threads per block: 256\n\n");
    
    printf("Execution Times:\n");
    printf("--------------------------------------------------------------------------------\n");
    
    // CPU Naive (Version 1)
    CpuTimer cpuTimer;
    startCpuTimer(&cpuTimer);
    pageRankCPU_Naive(graph.outLinks, graph.offsets, graph.outDegree, 
                      cpuNaive, graph.numPages, DAMPING_FACTOR, MAX_ITERATIONS);
    double cpuNaiveTime = stopCpuTimer(&cpuTimer);
    printf("  %-25s %10.3f ms\n", "CPU Naive", cpuNaiveTime);
    
    // CPU Optimized (Version 2)
    startCpuTimer(&cpuTimer);
    pageRankCPU_Optimized(graph.inLinks, graph.inOffsets, graph.outDegree,
                         cpuOpt, graph.numPages, DAMPING_FACTOR, MAX_ITERATIONS);
    double cpuOptTime = stopCpuTimer(&cpuTimer);
    printf("  %-25s %10.3f ms\n", "CPU Optimized", cpuOptTime);
    
    // GPU Naive (Version 3)
    float gpuNaiveTime = runGPU_PageRank("GPU Naive", pageRankKernel_Naive, 
                                         &graph, gpuNaive, false);
    
    // GPU Coalesced (Version 4)
    float gpuCoalescedTime = runGPU_PageRank("GPU Coalesced", pageRankKernel_Coalesced,
                                             &graph, gpuCoalesced, true);
    
    // GPU Shared Memory (Version 5)
    float gpuSharedTime = runGPU_PageRank("GPU Shared Memory", pageRankKernel_SharedMemory,
                                          &graph, gpuShared, true);
    
    // GPU Warp-Optimized (Version 6)
    float gpuWarpTime = runGPU_PageRank("GPU Warp-Optimized", pageRankKernel_WarpOptimized,
                                        &graph, gpuWarp, true);
    
    printf("\n");
    printf("================================================================================\n");
    printf("                          SPEEDUP ANALYSIS\n");
    printf("================================================================================\n\n");
    printf("Speedup vs. CPU Naive:\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("  %-25s %10.2fx\n", "CPU Optimized", cpuNaiveTime / cpuOptTime);
    printf("  %-25s %10.2fx\n", "GPU Naive", cpuNaiveTime / gpuNaiveTime);
    printf("  %-25s %10.2fx\n", "GPU Coalesced", cpuNaiveTime / gpuCoalescedTime);
    printf("  %-25s %10.2fx\n", "GPU Shared Memory", cpuNaiveTime / gpuSharedTime);
    printf("  %-25s %10.2fx\n", "GPU Warp-Optimized", cpuNaiveTime / gpuWarpTime);
    
    printf("\n");
    printf("================================================================================\n");
    printf("                          VERIFICATION\n");
    printf("================================================================================\n\n");
    
    // Verify all results match
    float maxDiff1 = verifyResults(cpuNaive, cpuOpt, graph.numPages);
    float maxDiff2 = verifyResults(cpuNaive, gpuNaive, graph.numPages);
    float maxDiff3 = verifyResults(cpuNaive, gpuCoalesced, graph.numPages);
    float maxDiff4 = verifyResults(cpuNaive, gpuShared, graph.numPages);
    float maxDiff5 = verifyResults(cpuNaive, gpuWarp, graph.numPages);
    
    printf("Maximum differences from CPU Naive:\n");
    printf("  CPU Optimized:       %e %s\n", maxDiff1, maxDiff1 < 1e-4 ? "✓" : "✗");
    printf("  GPU Naive:           %e %s\n", maxDiff2, maxDiff2 < 1e-4 ? "✓" : "✗");
    printf("  GPU Coalesced:       %e %s\n", maxDiff3, maxDiff3 < 1e-4 ? "✓" : "✗");
    printf("  GPU Shared Memory:   %e %s\n", maxDiff4, maxDiff4 < 1e-4 ? "✓" : "✗");
    printf("  GPU Warp-Optimized:  %e %s\n", maxDiff5, maxDiff5 < 1e-4 ? "✓" : "✗");
    
    // Print sample results for small graphs
    if (graphSize <= 10) {
        printRanks(cpuNaive, graph.numPages, "CPU Naive Results");
        printRanks(gpuWarp, graph.numPages, "GPU Warp-Optimized Results");
    }
    
    printf("\n");
    printf("================================================================================\n");
    
    // Cleanup
    free(cpuNaive);
    free(cpuOpt);
    free(gpuNaive);
    free(gpuCoalesced);
    free(gpuShared);
    free(gpuWarp);
    freeGraph(&graph);
    
    return 0;
}
