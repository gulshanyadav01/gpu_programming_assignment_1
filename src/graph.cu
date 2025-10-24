#include "../include/graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void createSampleGraph(Graph* graph) {
    graph->numPages = 7;
    graph->numEdges = 12;
    
    // Allocate memory for outgoing edges
    graph->outLinks = (int*)malloc(graph->numEdges * sizeof(int));
    graph->offsets = (int*)malloc((graph->numPages + 1) * sizeof(int));
    graph->outDegree = (int*)malloc(graph->numPages * sizeof(int));
    
    // Define graph structure (same as original)
    int links[] = {1, 2, 2, 0, 2, 4, 3, 5, 4, 5, 4, 0};
    for (int i = 0; i < graph->numEdges; i++) {
        graph->outLinks[i] = links[i];
    }
    
    int offs[] = {0, 2, 3, 4, 6, 8, 9, 12};
    for (int i = 0; i <= graph->numPages; i++) {
        graph->offsets[i] = offs[i];
    }
    
    for (int i = 0; i < graph->numPages; i++) {
        graph->outDegree[i] = graph->offsets[i + 1] - graph->offsets[i];
    }
    
    // Transpose graph for optimized versions
    transposeGraph(graph->outLinks, graph->offsets, graph->numPages, 
                   graph->numEdges, &graph->inLinks, &graph->inOffsets);
}

void generateRandomGraph(Graph* graph, int numPages, int avgDegree) {
    graph->numPages = numPages;
    graph->numEdges = numPages * avgDegree;
    
    // Allocate memory
    graph->outLinks = (int*)malloc(graph->numEdges * sizeof(int));
    graph->offsets = (int*)malloc((graph->numPages + 1) * sizeof(int));
    graph->outDegree = (int*)malloc(graph->numPages * sizeof(int));
    
    // Generate random edges
    srand(42); // Fixed seed for reproducibility
    int edgeIdx = 0;
    graph->offsets[0] = 0;
    
    for (int i = 0; i < numPages; i++) {
        int degree = avgDegree; // Could add randomness here
        graph->outDegree[i] = degree;
        
        for (int j = 0; j < degree && edgeIdx < graph->numEdges; j++) {
            int dest = rand() % numPages;
            // Avoid self-loops
            if (dest == i && numPages > 1) {
                dest = (dest + 1) % numPages;
            }
            graph->outLinks[edgeIdx++] = dest;
        }
        
        graph->offsets[i + 1] = edgeIdx;
    }
    
    graph->numEdges = edgeIdx; // Actual number of edges created
    
    // Transpose graph
    transposeGraph(graph->outLinks, graph->offsets, graph->numPages, 
                   graph->numEdges, &graph->inLinks, &graph->inOffsets);
}

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
    
    // Count incoming edges for each node
    int* inDegree = (int*)calloc(numPages, sizeof(int));
    for (int i = 0; i < numPages; i++) {
        for (int j = outOffsets[i]; j < outOffsets[i + 1]; j++) {
            inDegree[outLinks[j]]++;
        }
    }
    
    // Compute offsets for incoming edges
    (*inOffsets)[0] = 0;
    for (int i = 0; i < numPages; i++) {
        (*inOffsets)[i + 1] = (*inOffsets)[i] + inDegree[i];
    }
    
    // Fill incoming links
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

void freeGraph(Graph* graph) {
    if (graph->outLinks) free(graph->outLinks);
    if (graph->offsets) free(graph->offsets);
    if (graph->outDegree) free(graph->outDegree);
    if (graph->inLinks) free(graph->inLinks);
    if (graph->inOffsets) free(graph->inOffsets);
    
    memset(graph, 0, sizeof(Graph));
}

void printGraphStats(const Graph* graph) {
    printf("Graph Statistics:\n");
    printf("  Nodes: %d\n", graph->numPages);
    printf("  Edges: %d\n", graph->numEdges);
    printf("  Average degree: %.2f\n", (float)graph->numEdges / graph->numPages);
    
    // Find max degree
    int maxDegree = 0;
    for (int i = 0; i < graph->numPages; i++) {
        int degree = graph->offsets[i + 1] - graph->offsets[i];
        if (degree > maxDegree) maxDegree = degree;
    }
    printf("  Max degree: %d\n", maxDegree);
}
