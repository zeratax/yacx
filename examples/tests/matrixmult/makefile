C++FLAGS := -Wall -Wextra -pedantic -Werror -std=c++17 -fPIC
NVCCflags := -arch=sm_30 -Xptxas="-v" -std=c++14

TARGET = out
C_FILES := $(wildcard *.cpp)
OBJS := $(patsubst %.cpp, %.o, $(C_FILES))
KERNELS := $(wildcard *.cu)

$(TARGET): $(OBJS)
	g++ $(C++FLAGS) -o $(TARGET) $^ -lcuda

%.o: %.cpp
	g++ $(C++FLAGS) -c $< -o $@ -I $(CUDA_PATH)/include

%.cubin: %.cu
	nvcc $(NVCCflags) -c --cubin $< -o $@

clean:
	rm *.o $(TARGET) *.cubin
run:
	./$(TARGET)

.PHONY: run clean
