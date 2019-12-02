#pragma once

#include "../../include/cudaexecutor/JNIHandle.hpp"
#include "../../include/cudaexecutor/KernelArg.hpp"

namespace jni {
    class KernelArgJNI : cudaexecutor::JNIHandle {
        void* _hdata;
        cudaexecutor::KernelArg* kernelArg;

    public:
        KernelArgJNI(void* data, size_t size, bool download = false, bool copy = true, bool upload = false);
        ~KernelArgJNI();
        cudaexecutor::KernelArg* kernelArgPtr() { return kernelArg; }
        void* getHostData() { return _hdata; }
    };
} // namespace jni
