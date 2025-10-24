#ifndef GRAPH_H
#define GRAPH_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Graph data structure using Compressed Sparse Row (CSR) format
 */
typedef struct {
    int numPages;      // Number of nodes in the graph
    int numEdges;      // Number of edges in the graph
    int* outLinks;     // Destination nodes for each edge
    int* offsets;      // Starting index in outLinks for each node's edges
    int* outDegree;    // Number of outgoing edges for each node
    int* inLinks;      // Source nodes for incoming edges (transposed graph)
    int* inOffsets;    // Starting index in inLinks for each node's incoming edges
} Graph;

/**
 * Create a sample graph for testing
 */
void createSampleGraph(Graph* graph);

/**
 * Generate a random graph with specified size and average degree
 */
void generateRandomGraph(Graph* graph, int numPages, int avgDegree);

/**
 * Transpose graph to create incoming edge representation
 */
void transposeGraph(
    const int* outLinks,
    const int* outOffsets,
    int numPages,
    int numEdges,
    int** inLinks,
    int** inOffsets
);

/**
 * Free graph memory
 */
void freeGraph(Graph* graph);

/**
 * Print graph statistics
 */
void printGraphStats(const Graph* graph);

#ifdef __cplusplus
}
#endif

#endif // GRAPH_H
