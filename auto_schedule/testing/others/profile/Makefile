CXX := nvcc
TARGET := profile_autate
CUDNN_PATH := /usr/local/cuda-10.1
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64 -L/usr/local/lib
LIBDEVICE := --dont-use-profile -ldir $(CUDNN_PATH)/nvvm/libdevice
CXXFLAGS := -arch=sm_60 -std=c++11 -O2

all: conv

conv: $(TARGET).cu
	$(CXX) $(CXXFLAGS) $(LIBDEVICE) $(HEADERS) $(LIBS) $(TARGET).cu -o $(TARGET) 

.phony: clean

clean:
	rm $(TARGET) || echo -n ""