#include "IntArg.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/cudaexecutor/KernelArgs.hpp"

using cudaexecutor::KernelArg, jni::KernelArgJNI;

jobject Java_IntArg_createInternal (JNIEnv* env, jclass cls, jintArray jarray, jboolean jdownload){
    BEGIN_TRY
        CHECK_NULL(jarray)

        auto arrayPtr = env->GetIntArrayElements(jarray, NULL);
        auto arrayLength = env->GetArrayLength(jarray);

        CHECK_BIGGER(arrayLength, 0, "illegal array length")

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{arrayPtr, arrayLength * sizeof(jint), jdownload, true, arrayLength > 1};

        env->ReleaseIntArrayElements(jarray, arrayPtr, JNI_ABORT);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, kernelArgPtr);

        return obj;
    END_TRY("creating IntArg")
}

jobject Java_IntArg_createOutputInternal (JNIEnv* env, jclass cls, jint jarrayLength){
    BEGIN_TRY
        CHECK_BIGGER(jarrayLength, 0, "illegal array length")

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{NULL, static_cast<size_t> (jarrayLength) * sizeof(jint), true, false, true};

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, kernelArgPtr);

        return obj;
    END_TRY("creating IntArg")
}

jintArray Java_IntArg_asIntArray (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
        auto data = kernelArgJNIPtr->getHostData();
        auto dataSize = kernelArgJNIPtr->kernelArgPtr()->size();

        auto res = env->NewIntArray(dataSize / sizeof(jint));

        CHECK_NULL(res)

        env->SetIntArrayRegion(res, 0, dataSize / sizeof(jint),
                                   reinterpret_cast<const jint*>(data));
        return res;
    END_TRY("getting IntArg-content")
}

