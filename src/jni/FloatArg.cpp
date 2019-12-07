#include "FloatArg.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/cudaexecutor/KernelArgs.hpp"

using cudaexecutor::KernelArg, jni::KernelArgJNI;

jobject Java_FloatArg_createInternal (JNIEnv* env, jclass cls, jfloatArray jarray, jboolean jdownload){
    BEGIN_TRY
        CHECK_NULL(jarray)

        auto arrayPtr = env->GetFloatArrayElements(jarray, NULL);
        auto arrayLength = env->GetArrayLength(jarray);

        CHECK_BIGGER(arrayLength, 0, "illegal array length")

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{arrayPtr, arrayLength * sizeof(jfloat), jdownload, true, arrayLength > 1};

        env->ReleaseFloatArrayElements(jarray, arrayPtr, JNI_ABORT);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, kernelArgPtr);

        return obj;
    END_TRY("creating FloatArg")
}

jobject Java_FloatArg_createOutputInternal (JNIEnv* env, jclass cls, jint jarrayLength){
    BEGIN_TRY
        CHECK_BIGGER(jarrayLength, 0, "illegal array length")

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{NULL, static_cast<size_t> (jarrayLength) * sizeof(jfloat), true, false, true};

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, kernelArgPtr);

        return obj;
    END_TRY("creating FloatArg")
}

jfloatArray Java_FloatArg_asFloatArray (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
        auto data = kernelArgJNIPtr->getHostData();
        auto dataSize = kernelArgJNIPtr->kernelArgPtr()->size();

        auto res = env->NewFloatArray(dataSize / sizeof(jfloat));

        CHECK_NULL(res)

        env->SetFloatArrayRegion(res, 0, dataSize / sizeof(jfloat),
                                   reinterpret_cast<const jfloat*>(data));
        return res;
    END_TRY("getting FloatArg-content")
}

