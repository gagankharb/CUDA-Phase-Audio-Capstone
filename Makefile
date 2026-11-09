CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CXX       ?= g++
TARGET    := build/phase_audio_cuda
SRC       := src/main.cu
CUDA_ARCH ?= -gencode arch=compute_75,code=sm_75

NVCCFLAGS := -O3 -std=c++17 -lineinfo $(CUDA_ARCH) -I$(CUDA_HOME)/include
LDFLAGS   := -L$(CUDA_HOME)/lib64 -lcufft

.PHONY: all clean run demo
all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -rf build output

demo: all
	python3 scripts/generate_demo_wav.py --output data/demo.wav
	mkdir -p output
	./$(TARGET) --input data/demo.wav --output output/demo_gpu.csv --backend gpu

run: demo
