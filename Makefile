# PageRank CUDA Implementation Makefile
# Author: GPU Programming Project
# Date: October 2025

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++11 -Xcompiler -Wall
ARCH = -arch=sm_75  # Adjust based on your GPU (sm_60, sm_70, sm_75, sm_80, sm_86, sm_89)

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
BIN_DIR = bin

# Source files
SOURCES = $(SRC_DIR)/main.cu $(SRC_DIR)/graph.cu $(SRC_DIR)/utils.cu $(SRC_DIR)/kernels.cu

# Target executable
TARGET = $(BIN_DIR)/pagerank

# Default target
all: directories $(TARGET)

# Create necessary directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# Build main executable
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -I$(INCLUDE_DIR) -o $@ $^
	@echo "Build successful! Executable: $(TARGET)"

# Run with sample graph (7 nodes)
run: $(TARGET)
	./$(TARGET)

# Run with larger graphs for benchmarking
benchmark: $(TARGET)
	@echo "\n=== Benchmarking with different graph sizes ===\n"
	@echo "Running 1,000 nodes..."
	./$(TARGET) 1000
	@echo "\n\nRunning 5,000 nodes..."
	./$(TARGET) 5000
	@echo "\n\nRunning 10,000 nodes..."
	./$(TARGET) 10000

# Test with original pagerank.cu for comparison
original:
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -o $(BIN_DIR)/pagerank_original pagerank.cu
	./$(BIN_DIR)/pagerank_original

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "Clean complete"

# Clean and rebuild
rebuild: clean all

# Display help information
help:
	@echo "PageRank CUDA Implementation - Makefile Commands"
	@echo "=================================================="
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build the complete PageRank suite (default)"
	@echo "  run          - Build and run with sample 7-node graph"
	@echo "  benchmark    - Run comprehensive benchmarks (1K, 5K, 10K nodes)"
	@echo "  original     - Build and run original pagerank.cu"
	@echo "  clean        - Remove build artifacts"
	@echo "  rebuild      - Clean and rebuild from scratch"
	@echo "  help         - Display this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make                 # Build everything"
	@echo "  make run             # Run with default graph"
	@echo "  ./bin/pagerank 1000  # Run with 1000-node graph"
	@echo "  make benchmark       # Run full benchmark suite"
	@echo ""
	@echo "Configuration:"
	@echo "  Compute Architecture: $(ARCH)"
	@echo "  Compiler: $(NVCC)"
	@echo ""
	@echo "Note: Adjust ARCH in Makefile to match your GPU:"
	@echo "  - sm_60: Pascal (GTX 10 series, P100)"
	@echo "  - sm_70: Volta (V100)"
	@echo "  - sm_75: Turing (RTX 20 series, T4)"
	@echo "  - sm_80: Ampere (A100, RTX 30 series)"
	@echo "  - sm_86: Ampere (RTX 30 series mobile)"
	@echo "  - sm_89: Ada (RTX 40 series)"

# Phony targets
.PHONY: all directories run benchmark original clean rebuild help
