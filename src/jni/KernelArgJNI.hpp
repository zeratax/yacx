#pragma once

#include "../../include/yacx/JNIHandle.hpp"
#include "../../include/yacx/KernelArgs.hpp"

namespace jni {
    class KernelArgJNI : yacx::JNIHandle {
        void* _hdata;
        yacx::KernelArg* kernelArg;

    public:
        KernelArgJNI(void* data, size_t size, bool download = false, bool copy = true, bool upload = false);
        ~KernelArgJNI();
        yacx::KernelArg* kernelArgPtr() { return kernelArg; }
        void* getHostData() { return _hdata; }
    };
} // namespace jni
