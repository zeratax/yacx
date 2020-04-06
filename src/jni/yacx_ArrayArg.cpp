#include "yacx_ArrayArg.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/yacx/KernelArgs.hpp"

using yacx::KernelArg, jni::KernelArgJNI, jni::KernelArgJNISlice;

jlong Java_yacx_ArrayArg_createOutputInternal(JNIEnv* env, jclass, jlong jNumberBytes){
    BEGIN_TRY
        CHECK_BIGGER(jNumberBytes, 0, "illegal array length", 0)

        KernelArgJNI* kernelArgJNIPtr = new KernelArgJNI{NULL, static_cast<size_t> (jNumberBytes), true, false, true, "*"};

    	return reinterpret_cast<jlong> (kernelArgJNIPtr);
    END_TRY_R("creating output argument", 0)
}

jlong Java_yacx_ArrayArg_getSize(JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
    	CHECK_NULL(kernelArgJNIPtr, 0)
        auto dataSize = kernelArgJNIPtr->kernelArgPtr()->size();

        return dataSize;
    END_TRY_R("getting size of an argument", 0)
}

jboolean Java_yacx_ArrayArg_isDownload(JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
    	CHECK_NULL(kernelArgJNIPtr, false)
        auto download = kernelArgJNIPtr->kernelArgPtr()->isDownload();

        return download;
    END_TRY_R("getting download status", false)
}

void Java_yacx_ArrayArg_setDownload(JNIEnv* env, jobject obj, jboolean jdownload){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
    	CHECK_NULL(kernelArgJNIPtr, )
        auto kernelArgPtr = kernelArgJNIPtr->kernelArgPtr();

        kernelArgPtr->setDownload(jdownload);
    END_TRY("setting download status")
}

jboolean Java_yacx_ArrayArg_isUpload(JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
    	CHECK_NULL(kernelArgJNIPtr, false)
        auto upload = kernelArgJNIPtr->kernelArgPtr()->isCopy();

        return upload;
    END_TRY_R("getting upload status", false)
}

void Java_yacx_ArrayArg_setUpload(JNIEnv* env, jobject obj, jboolean jupload){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
    	CHECK_NULL(kernelArgJNIPtr, )
        auto kernelArgPtr = kernelArgJNIPtr->kernelArgPtr();

        kernelArgPtr->setCopy(jupload);
    END_TRY("setting upload status")
}

jlong Java_yacx_ArrayArg_sliceInternal(JNIEnv* env, jobject obj, jlong jstart, jlong jend){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
    	CHECK_NULL(kernelArgJNIPtr, 0)

        CHECK_BIGGER(jstart, -1, "illegal indices for slice", 0)
        CHECK_BIGGER(jend, jstart, "illegal indices for slice", 0)
        CHECK_BIGGER(kernelArgJNIPtr->kernelArgPtr()->size(), jend-1, "illegal indices for slice", 0)

        KernelArgJNISlice* newkernelArgJNIPtr = new KernelArgJNISlice{static_cast<size_t> (jstart),
            static_cast<size_t> (jend), kernelArgJNIPtr};

        return reinterpret_cast<jlong> (newkernelArgJNIPtr);
    END_TRY_R("slicing an array argument", 0)
}