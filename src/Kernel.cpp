//
// Created by hadis on 11/22/19.
//

#include <cudaexecutor/Exception.hpp>
#include <utility>
#include "cudaexecutor/Kernel.hpp"

using cudaexecutor::Kernel, cudaexecutor::loglevel;

Kernel::Kernel(char* ptx, std::vector<std::string> template_parameters, std::string kernel_name,
               std::string name_expression, nvrtcProgram prog)
        :  _ptx(ptx), _template_parameters(std::move(template_parameters)), _kernel_name(std::move(kernel_name)),
          _name_expression(std::move(name_expression)), _prog(prog) {
    logger(loglevel::DEBUG) << "created Kernel " << _kernel_name;
}

Kernel &Kernel::configure(dim3 grid, dim3 block) {
    logger(loglevel::DEBUG) << "configuring Kernel with grid: " << grid.x << ", "
                            << grid.y << ", " << grid.z << " and block "
                            << block.x << ", " << block.y << ", " << block.z;
    _grid = grid;
    _block = block;
    return *this;
}

Kernel &Kernel::launch(std::vector<ProgramArg> args) {
    logger(loglevel::DEBUG) << "launching Kernel";

    // check if device already initialised
    CUDA_SAFE_CALL(cuDeviceGet(&_cuDevice, 0));

    CUDA_SAFE_CALL(cuCtxCreate(&_context, 0, _cuDevice));
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&_module, _ptx, 0, nullptr, nullptr));

    logger(loglevel::DEBUG) << "uploading arguemnts";
    void *kernel_args[args.size()];
    int i{0};
    for (auto &arg : args) {
        arg.upload();
        kernel_args[i++] = arg.content();
    }

    // lowered name
    const char *name = _kernel_name.c_str();
    if (!_template_parameters.empty()) {
        logger(loglevel::DEBUG) << "getting lowered name for function";
        NVRTC_SAFE_CALL(
                nvrtcGetLoweredName(_prog, _name_expression.c_str(), &name))
    }
    CUDA_SAFE_CALL(cuModuleGetFunction(&_kernel, _module, name));

    // launch the program

    logger(loglevel::INFO) << "launching " << name << "<" << _name_expression
                           << ">";
    CUDA_SAFE_CALL(cuLaunchKernel(_kernel, // function from program
                                  _grid.x, _grid.y, _grid.z,    // grid dim
                                  _block.x, _block.y, _block.z, // block dim
                                  0, nullptr,             // shared mem and stream
                                  kernel_args, nullptr)); // arguments
    CUDA_SAFE_CALL(cuCtxSynchronize());
    logger(loglevel::INFO) << "done!";

    // download results to host
    logger(loglevel::DEBUG) << "downloading arguments";
    for (auto &arg : args)
        arg.download();

    logger(loglevel::DEBUG) << "freeing resources";
    CUDA_SAFE_CALL(cuModuleUnload(_module));
    CUDA_SAFE_CALL(cuCtxDestroy(_context));

    return *this;
}


