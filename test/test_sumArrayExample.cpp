#include "yacx/main.hpp"
#include <catch2/catch.hpp>

using yacx::Kernel, yacx::Source, yacx::KernelArg, yacx::Options, yacx::Device, yacx::type_of;


TEST_CASE("sumArray", "[example_program]") {

    Device device;
        Options options{yacx::options::GpuArchitecture(device),
                        yacx::options::FMAD(false)};
        options.insert("--std", "c++14");
        Source source{"extern \"C\" __global__ void sumArrayOnGPU(float *A, float *B, float* C){\n"
                        "int i_inBlock=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.y*blockDim.x;\n"
                        "int blockID= blockIdx.x;\n"
                        "int i = i_inBlock + blockID*(blockDim.x*blockDim.y*blockDim.z);\n"
                        "C[i]=A[i]+B[i];\n"
                        "}\n"
        };

  SECTION("SUM_ARRAY WITH 1 and fixed values") {
        // set up data size of vectors
        int nElem = 1; 
        //malloc host memory
        size_t nBytes = nElem * sizeof(float);
        float *h_A, *h_B, *hostRef, *gpuRef;
        h_A = (float*) malloc(nBytes);
        h_B = (float*) malloc(nBytes);
        hostRef = (float*) malloc(nBytes);
        gpuRef = (float*) malloc(nBytes);
        //initialize data at host side
        //initialData(h_A, nElem);
            // generate different seed for random number
            for (int i=0; i<nElem; i++) {
                h_A[i] = 42;
            }
        //initialData(h_B, nElem);
            // generate different seed for random number
            for (int i=0; i<nElem; i++) {
                h_B[i] = 18;
            }
    
        std::vector<KernelArg> args;
        args.emplace_back(KernelArg{h_A, nBytes, false});
        args.emplace_back(KernelArg{h_B, nBytes, false});
        args.emplace_back(KernelArg{gpuRef, nBytes, true});

        dim3 block (1, 1 , 1);
        dim3 grid (1);
        Kernel k = source.program("sumArrayOnGPU")
            .compile(options)
            .configure(grid, block);
            k.launch(args, device);
    

        //add vector at host side for result checks
        for (int idx=0; idx<nElem; idx++)
            hostRef[idx] = h_A[idx] + h_B[idx];

        //check device results
         double epsilon = 1.0E-8;
        bool match = 1;
        for(int i=0;i<nElem;++i){
            if(abs(hostRef[i]-gpuRef[i]) > epsilon){
            match=0;
            char *error_string;
            asprintf(&error_string, "Arrays do not match! \n->host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            FAIL(error_string);
            break;
            }
        }
        free(h_A);
        free(h_B);
        free(hostRef);
        free(gpuRef);
  }


    SECTION("SUM_ARRAY WITH 1024 and fixed values") {
        // set up data size of vectors
        int nElem = 1024; 
        //malloc host memory
        size_t nBytes = nElem * sizeof(float);
        float *h_A, *h_B, *hostRef, *gpuRef;
        h_A = (float*) malloc(nBytes);
        h_B = (float*) malloc(nBytes);
        hostRef = (float*) malloc(nBytes);
        gpuRef = (float*) malloc(nBytes);
        //initialize data at host side
        //initialData(h_A, nElem);
            // generate different seed for random number
            for (int i=0; i<nElem; i++) {
                h_A[i] = 42;
            }
        //initialData(h_B, nElem);
            // generate different seed for random number
            for (int i=0; i<nElem; i++) {
                h_B[i] = 18;
            }
    
        std::vector<KernelArg> args;
        args.emplace_back(KernelArg{h_A, nBytes, false});
        args.emplace_back(KernelArg{h_B, nBytes, false});
        args.emplace_back(KernelArg{gpuRef, nBytes, true});

        dim3 block (32, 8 , 2);
        int blockSize = block.x*block.y*block.z;
        dim3 grid ((nElem+blockSize-1)/blockSize);
        Kernel k = source.program("sumArrayOnGPU")
            .compile(options)
            .configure(grid, block);
            k.launch(args, device);
    

        //add vector at host side for result checks
        for (int idx=0; idx<nElem; idx++)
            hostRef[idx] = h_A[idx] + h_B[idx];

        //check device results
         double epsilon = 1.0E-8;
        bool match = 1;
        for(int i=0;i<nElem;++i){
            if(abs(hostRef[i]-gpuRef[i]) > epsilon){
            match=0;
            char *error_string;
            asprintf(&error_string, "Arrays do not match! \n->host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            FAIL(error_string);
            break;
            }
        }
        free(h_A);
        free(h_B);
        free(hostRef);
        free(gpuRef);
  }







  SECTION("SUM_ARRAY WITH 1024 and random values") {
        
        // set up data size of vectors
        int nElem = 1024; 
        //malloc host memory
        size_t nBytes = nElem * sizeof(float);
        float *h_A, *h_B, *hostRef, *gpuRef;
        h_A = (float*) malloc(nBytes);
        h_B = (float*) malloc(nBytes);
        hostRef = (float*) malloc(nBytes);
        gpuRef = (float*) malloc(nBytes);
        //initialize data at host side
        //initialData(h_A, nElem);
            // generate different seed for random number
            time_t t;
            srand((unsigned) time(&t));
            for (int i=0; i<nElem; i++) {
                h_A[i] = (float)( rand() & 0xFF )/10.0f;
            }
        //initialData(h_B, nElem);
            // generate different seed for random number
            srand((unsigned) time(&t));
            for (int i=0; i<nElem; i++) {
                h_B[i] = (float)( rand() & 0xFF )/10.0f;
            }
    
        std::vector<KernelArg> args;
        args.emplace_back(KernelArg{h_A, nBytes, false});
        args.emplace_back(KernelArg{h_B, nBytes, false});
        args.emplace_back(KernelArg{gpuRef, nBytes, true});

        dim3 block (32, 8 , 2);
        int blockSize = block.x*block.y*block.z;
        dim3 grid ((nElem+blockSize-1)/blockSize); 
        Kernel k = source.program("sumArrayOnGPU")
            .compile(options)
            .configure(grid, block);
            k.launch(args, device);
    

        //add vector at host side for result checks
        for (int idx=0; idx<nElem; idx++)
            hostRef[idx] = h_A[idx] + h_B[idx];

        //check device results
         double epsilon = 1.0E-8;
        bool match = 1;
        for(int i=0;i<nElem;++i){
            if(abs(hostRef[i]-gpuRef[i]) > epsilon){
            match=0;
            char *error_string;
            asprintf(&error_string, "Arrays do not match! \n->host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            FAIL(error_string);
            break;
            }
        }
        free(h_A);
        free(h_B);
        free(hostRef);
        free(gpuRef);
  }
}