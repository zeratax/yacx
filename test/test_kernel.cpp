//Here, the sources of kernels are tested.
#include "yacx/Headers.hpp"
#include "yacx/KernelArgs.hpp"
#include "yacx/Kernel.hpp"
#include "yacx/Source.hpp"

#include <catch2/catch.hpp>
#include <iostream>

using yacx::Kernel, yacx::KernelTime, yacx::KernelArg, yacx::Source, 
yacx::Header, yacx::Headers;

TEST_CASE("The kernel - source code will be tested under the following conditions."){
    //A. Preparing the input for the kernel-compilation using source
    int datasize{10};
    int *hX = new int[10]{1,2,3,4,5,6,7,8,9,10};
    int *hY = new int[10]{6,7,8,9,10,11,12,13,14,15};
    int *hOut = new int[10]{7,9,11,13,15,1,2,3,4,5};
    size_t bufferSize = 10*sizeof(int);
    
    std::vector<KernelArg> args;
    args.emplace_back(KernelArg{hX, bufferSize});
    args.emplace_back(KernelArg{hY, bufferSize});
    args.emplace_back(KernelArg{hOut, bufferSize, true});
    args.emplace_back(KernelArg(&datasize));

    Headers headers;
    headers.insert(Header{"cuda_runtime.h"});

    //A2. Preparing the output for kernel-compilation
    int *hostCompareOutput = new int[10]{7,9,11,13,15,17,19,21,23,25};
    
    //B1. Configure a kernel using the following block and grid dimensions
    SECTION("1. The created kernel is configured."){     
        Source source{"#include \"cuda_runtime.h\"\n"
        "extern \"C\"\n"
        "__global__ void cuda_add(int *x, int *y, int *out, int "
        "datasize) {\n"
        " int i = threadIdx.x;\n"
        " out[i] = x[i] + y[i];\n"
        "}", headers};

        dim3 grid_test(1);
        dim3 block_test(5);
        int counter = 1;
        
        source.program("cuda_add").compile().configure(grid_test, block_test).launch(args);

        //B1A. Comparing the results
        for (int i=0;i<5;i++){
            REQUIRE(hOut[i] == hostCompareOutput[i]);
        }

        for (int j=5;j<10;j++){
            REQUIRE(hOut[j] == counter);
            counter++;
        }
    }
    
    //B2. Lauching a kernel using the following block and grid dimensions
    SECTION("2. The created kernel is launched."){     
        Source source{"#include \"cuda_runtime.h\"\n"
        "extern \"C\"\n"
        "__global__ void cuda_add_launching(int *x, int *y, int *out, int "
        "datasize) {\n"
        " int i = threadIdx.x;\n"
        " out[i] = x[i] + y[i];\n"
        "}", headers};

        dim3 grid_test(1);
        dim3 block_test(10);
        
        KernelTime kernel_lauching = source.program("cuda_add_launching").compile().configure(grid_test, block_test).launch(args);

        //B2A. Comparing the results
        //B2A1. Checking for consistencies between given output constraints and host results
        for (int i=0;i<10;i++){
            REQUIRE(hOut[i] == hostCompareOutput[i]);
        }

        //B2A2. Ensuring that the kernel is launched
        REQUIRE(kernel_lauching.sum>0);
        REQUIRE(kernel_lauching.download>0);
        REQUIRE(kernel_lauching.launch>0);
    }
}
