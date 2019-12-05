#pragma once

#include "../../include/cudaexecutor/JNIHandle.hpp"
#include "cudaexecutor/KernelArgs.hpp"

namespace jni {
    class ProgramArgJNI : cudaexecutor::JNIHandle {
        void* _hdata;
        cudaexecutor::ProgramArg* programArg;

    public:
        ProgramArgJNI(void* data, size_t size, bool download = false, bool copy = true, bool upload = false);
        ~ProgramArgJNI();
        cudaexecutor::ProgramArg* programArgPtr() { return programArg; }
        void* getHostData() { return _hdata; }
    };
} // namespace jni
