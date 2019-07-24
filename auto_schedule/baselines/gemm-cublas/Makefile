CXX := nvcc
TARGET := cublas_gemm
CUBLAS_PATH := /usr/local/cuda-10.1
HEADERS := -I $(CUBLAS_PATH)/include
LIBS := -L $(CUBLAS_PATH)/lib64 -L/usr/local/lib
LIBDEVICE := --dont-use-profile -ldir $(CUBLAS_PATH)/nvvm/libdevice
CXXFLAGS := -arch=sm_70 -std=c++11 -O2

all: conv

conv: $(TARGET).cu
	$(CXX) $(CXXFLAGS) $(LIBDEVICE) $(HEADERS) $(LIBS) $(TARGET).cu -o $(TARGET) \
	-lcublas

.phony: clean

clean:
	rm $(TARGET) || echo -n ""