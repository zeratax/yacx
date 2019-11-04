# von André Hodapp

#-lcuda ist notwendig damit mit Cuda gelinkt wird
C++FLAGS := -Wall -Wextra -pedantic -Werror -std=c++14 -fPIC 
out: driverApiProg.o #matSumKernel.cubin
	g++ $(C++FLAGS) -o out driverApiProg.o -lcuda

driverApiProg.o: driverApiProg.cpp
	g++ $(C++FLAGS) -c driverApiProg.cpp -I $(CUDA_PATH)/include

#nvcc -arch=sm_30 -Xptxas="-v" --cubin matSumKernel.cu
#siehe: https://stackoverflow.com/questions/18844976/cuda-error-invalid-image-during-cumoduleload
matSumKernel.cubin: matSumKernel.cu
	nvcc -arch=sm_30 -Xptxas="-v" --cubin matSumKernel.cu

clean:
	rm *.o driverApiProg *.cubin
run:
	./driverApiProg
