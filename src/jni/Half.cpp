#include "Half.hpp"

#include "../../include/yacx/Devices.hpp"
#include "../../include/yacx/Kernel.hpp"
#include "../../include/yacx/Program.hpp"
#include "../../include/yacx/Source.hpp"

#include <stdlib.h>
#include <vector>

using yacx::Source, yacx::KernelArg, yacx::Kernel, yacx::Device, yacx::Devices, yacx::Options;

unsigned int maxGridSize = 0;
yacx::Kernel* kernelFtoH = NULL;
yacx::Kernel* kernelHtoF = NULL;
yacx::Device* device = NULL;

void initKernel(){
    std::vector<Device*> devices = Devices::findDevices([](Device* device){ return device->major_version() >= 6; });
    if (devices.empty()){
        throw std::invalid_argument("no CUDA-device with computeversion >= 6 found for conversion from/to halfs");
    }
    device = devices[0];

    dim3 grid;
    device->max_grid_dim(&grid);
    maxGridSize = grid.x;

    Options options;
    options.insert("--gpu-architecture=compute_60");

    Source source{
            "#include <cuda_fp16.h>\n"
                "extern \"C\" __global__\n"
        		"void floatToHalf(float* floats, half* out, unsigned int n) {\n"
        		"  for (unsigned int i = blockIdx.x; i < n; i += gridDim.x){\n"
        		"    out[i] = __float2half(floats[i]);\n"
        		"  }\n"
        		"}"};

    kernelFtoH = new Kernel{source.program("floatToHalf").compile(options)};

    Source source2{
        "#include <cuda_fp16.h>\n"
                "extern \"C\" __global__\n"
        		"void halfToFloat(half* halfs, float* out, unsigned int n) {\n"
        		"  for (unsigned int i = blockIdx.x; i < n; i += gridDim.x){\n"
        		"    out[i] = __half2float(halfs[i]);\n"
        		"  }\n"
        		"}"};

    kernelHtoF = new Kernel{source2.program("halfToFloat").compile(options)};
}

void yacx::convertFtoH(void* floats, void* halfs, unsigned int length){
    if (kernelFtoH == NULL){
        initKernel();
    }

    std::vector<KernelArg> args;
    args.emplace_back(KernelArg{floats, length*sizeof(float), false, true, true});
    args.emplace_back(KernelArg{halfs, length*sizeof(float)/2, true, false, true});
    args.emplace_back(KernelArg{const_cast<unsigned int*>(&length)});

    dim3 grid(length < maxGridSize ? length/1024+1 : maxGridSize);
    dim3 block(1);

    kernelFtoH->configure(grid, block).launch(args, device);
}

void yacx::convertHtoF(void* halfs, void* floats, unsigned int length){
    if (kernelHtoF == NULL){
        initKernel();
    }

    std::vector<KernelArg> args;
    args.emplace_back(KernelArg{halfs, length*sizeof(float)/2, false, true, true});
    args.emplace_back(KernelArg{floats, length*sizeof(float), true, false, true});
    args.emplace_back(KernelArg{const_cast<unsigned int*>(&length)});

    dim3 grid(length < maxGridSize ? length/1024+1 : maxGridSize);
    dim3 block(1);

    kernelHtoF->configure(grid, block).launch(args, device);
}
