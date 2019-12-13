#include "DoubleArg.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/yacx/KernelArgs.hpp"

using yacx::KernelArg, jni::KernelArgJNI;

jobject Java_DoubleArg_createInternal (JNIEnv* env, jclass cls, jdoubleArray jarray, jboolean jdownload){
    BEGIN_TRY
        CHECK_NULL(jarray, NULL)

        auto arrayPtr = env->GetDoubleArrayElements(jarray, NULL);
        auto arrayLength = env->GetArrayLength(jarray);

        CHECK_BIGGER(arrayLength, 0, "illegal array length", NULL)

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{arrayPtr, arrayLength * sizeof(jdouble), jdownload, true, arrayLength > 1};

        env->ReleaseDoubleArrayElements(jarray, arrayPtr, JNI_ABORT);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, kernelArgPtr);

        return obj;
    END_TRY("creating DoubleArg")
}

jobject Java_DoubleArg_createOutputInternal (JNIEnv* env, jclass cls, jint jarrayLength){
    BEGIN_TRY
        CHECK_BIGGER(jarrayLength, 0, "illegal array length", NULL)

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{NULL, static_cast<size_t> (jarrayLength) * sizeof(jdouble), true, false, true};

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, kernelArgPtr);

        return obj;
    END_TRY("creating DoubleArg")
}

jdoubleArray Java_DoubleArg_asDoubleArray (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
        auto data = kernelArgJNIPtr->getHostData();
        auto dataSize = kernelArgJNIPtr->kernelArgPtr()->size();

        auto res = env->NewDoubleArray(dataSize / sizeof(jdouble));

        CHECK_NULL(res, NULL)

        env->SetDoubleArrayRegion(res, 0, dataSize / sizeof(jdouble),
                                   reinterpret_cast<const jdouble*>(data));
        return res;
    END_TRY("getting DoubleArg-content")
}

