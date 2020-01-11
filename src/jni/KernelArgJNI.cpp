#include "KernelArgJNI.hpp"

#include <cstring>
#include <stdio.h>

using jni::KernelArgJNI, jni::KernelArgJNISlice, yacx::KernelArg, std::shared_ptr;

KernelArgJNI::KernelArgJNI(void* const data, size_t size, bool download, bool copy, bool upload) {
    std::shared_ptr<void> hdata(malloc(size), free);
    m_hdata = hdata;

    if (data)
        std::memcpy(hdata.get(), data, size);

    m_kernelArg = new KernelArg{hdata.get(), size, download, copy, upload};
}

KernelArgJNI::~KernelArgJNI() {
    delete m_kernelArg;
}

KernelArgJNISlice::KernelArgJNISlice(size_t start, size_t end, KernelArgJNI* arg) :
    KernelArgJNI(arg->m_hdata, new KernelArg{reinterpret_cast<char*> (arg->getHostData()) + start,
        end-start, arg->kernelArgPtr()->isDownload(), arg->kernelArgPtr()->isCopy(), true}),
    m_offset(static_cast<char*> (arg->getHostData()) - static_cast<char*> (arg->m_hdata.get())
        + start) {}
