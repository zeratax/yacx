#include "ShortArg.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/cudaexecutor/KernelArgs.hpp"

using cudaexecutor::KernelArg, jni::KernelArgJNI;

jobject Java_ShortArg_createInternal (JNIEnv* env, jclass cls, jshortArray jarray, jboolean jdownload){
    BEGIN_TRY
        CHECK_NULL(jarray)

        auto arrayPtr = env->GetShortArrayElements(jarray, NULL);
        auto arrayLength = env->GetArrayLength(jarray);

        CHECK_BIGGER(arrayLength, 0, "illegal array length")

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{arrayPtr, arrayLength * sizeof(jshort), jdownload, true, arrayLength > 1};

        env->ReleaseShortArrayElements(jarray, arrayPtr, JNI_ABORT);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, kernelArgPtr);

        return obj;
    END_TRY("creating ShortArg")
}

jobject Java_ShortArg_createOutputInternal (JNIEnv* env, jclass cls, jint jarrayLength){
    BEGIN_TRY
        CHECK_BIGGER(jarrayLength, 0, "illegal array length")

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{NULL, static_cast<size_t> (jarrayLength) * sizeof(jshort), true, false, true};

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, kernelArgPtr);

        return obj;
    END_TRY("creating ShortArg")
}

jshortArray Java_ShortArg_asShortArray (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
        auto data = kernelArgJNIPtr->getHostData();
        auto dataSize = kernelArgJNIPtr->kernelArgPtr()->size();

        auto res = env->NewShortArray(dataSize / sizeof(jshort));

        CHECK_NULL(res)

        env->SetShortArrayRegion(res, 0, dataSize / sizeof(jshort),
                                   reinterpret_cast<const jshort*>(data));
        return res;
    END_TRY("getting ShortArg-content")
}

