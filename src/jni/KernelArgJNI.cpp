#include "KernelArgJNI.hpp"

#include "../../include/yacx/Init.hpp"
#include "../../include/yacx/Exception.hpp"
#include <cstring>
#include <stdio.h>

using jni::KernelArgJNI, jni::KernelArgJNISlice, yacx::KernelArg, std::shared_ptr, jni::detail::HDataMem;

HDataMem::HDataMem(size_t size){
    yacx::detail::initCtx();
    CUDA_SAFE_CALL(cuMemAllocHost(&m_hdata, size));
}

HDataMem::~HDataMem(){
    CUDA_SAFE_CALL(cuMemFreeHost(m_hdata));
}

KernelArgJNI::KernelArgJNI(size_t size, bool download, bool copy, bool upload, std::string type) : m_type(type) {
    m_hdata = std::make_shared<HDataMem>(size);

    m_kernelArg = new KernelArg{getHostData(), size, download, copy, upload};
}

KernelArgJNI::~KernelArgJNI() {
    delete m_kernelArg;
}

KernelArgJNISlice::KernelArgJNISlice(size_t start, size_t end, KernelArgJNI* arg) :
    KernelArgJNI(arg->m_hdata, new KernelArg{reinterpret_cast<char*> (arg->getHostData()) + start,
        end-start, arg->kernelArgPtr()->isDownload(), arg->kernelArgPtr()->isCopy(), true},
        arg->getType()),
        m_offset(static_cast<char*> (arg->getHostData()) - static_cast<char*> (arg->m_hdata.get()->get())
        + start) {}
