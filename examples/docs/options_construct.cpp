#include "cudaexecutor/Device.hpp"
#include "cudaexecutor/Options.hpp"

Device device;
Options options{cudaexecutor::options::GpuArchitecture(device),
                cudaexecutor::options::FMAD(false)};
options.insertOption(cudaexecutor::options::Fast_Math{true});
options.insert("--std", "c++14");