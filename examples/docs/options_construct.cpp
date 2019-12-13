#include "yacx/Device.hpp"
#include "yacx/Options.hpp"

Device device;
Options options{yacx::options::GpuArchitecture(device),
                yacx::options::FMAD(false)};
options.insertOption(yacx::options::Fast_Math{true});
options.insert("--std", "c++14");