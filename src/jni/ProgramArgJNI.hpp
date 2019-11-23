#pragma once

#include "../../include/cudaexecutor/JNIHandle.hpp"
#include "../../include/cudaexecutor/ProgramArg.hpp"

namespace jni {
    class ProgramArgJNI : cudaexecutor::JNIHandle {
        void* _hdata;
        cudaexecutor::ProgramArg* programArg;

    public:
        ProgramArgJNI(void* data, size_t size, bool download, bool copy, bool upload);
        ProgramArgJNI(void* data, size_t size) : ProgramArgJNI{data, size, false, false, false} {};
        ~ProgramArgJNI();
        cudaexecutor::ProgramArg* programArgPtr() {return programArg;}
    };
} // namespace jni
