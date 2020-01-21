#include "yacx_PaddingArg.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/yacx/KernelArgs.hpp"

using yacx::KernelArg, yacx::KernelArgMatrixPadding, jni::KernelArgJNI;

jobject Java_yacx_PaddingArg_createMatrixPaddingInternal(JNIEnv* env, jclass cls, jobject jKernelArg,
    jint jcolumnsArg, jint jrowsArg, jint jcolumnsNew, jint jrowsNew, jint jpaddingValue,
    jboolean jshortElements) {
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, jKernelArg);
        auto kernelArgPtr = kernelArgJNIPtr->kernelArgPtr();
        CHECK_NULL(kernelArgJNIPtr, NULL)

        CHECK_BIGGER(jcolumnsArg, 0, "illegal matrix dimension columnsArg", NULL)
        CHECK_BIGGER(jrowsArg, 0, "illegal matrix dimension rowsArg", NULL)
        CHECK_BIGGER(jcolumnsNew, jcolumnsArg-1, "illegal matrix dimension columnsNew", NULL)
        CHECK_BIGGER(jrowsNew, jrowsArg-1, "illegal matrix dimension rowsNew", NULL)
        CHECK_BIGGER(jcolumnsNew+jrowsNew, jcolumnsArg+jrowsArg, "illegal matrix dimensions: new matrix dimensions are current matrix dimensions", NULL)

        KernelArgMatrixPadding* newKernelArgPtr = new KernelArgMatrixPadding{kernelArgJNIPtr->getHostData(),
            kernelArgPtr->size(), jrowsNew, jcolumnsNew, jrowsArg, jcolumnsArg, jpaddingValue,
            jshortElements, kernelArgPtr->isDownload()};

        KernelArgJNI* newKernelArgJNIPtr = new KernelArgJNI{kernelArgJNIPtr->getHostDataSharedPointer(),
            newKernelArgPtr};
        return createJNIObject(env, cls, newKernelArgJNIPtr);
    END_TRY_R("create PaddinArg", NULL)
}