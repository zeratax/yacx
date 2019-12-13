#include "LongArg.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/yacx/KernelArgs.hpp"

using yacx::KernelArg, jni::KernelArgJNI;

jobject Java_LongArg_createInternal (JNIEnv* env, jclass cls, jlongArray jarray, jboolean jdownload){
    BEGIN_TRY
        CHECK_NULL(jarray, NULL)

        auto arrayPtr = env->GetLongArrayElements(jarray, NULL);
        auto arrayLength = env->GetArrayLength(jarray);

        CHECK_BIGGER(arrayLength, 0, "illegal array length", NULL)

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{arrayPtr, arrayLength * sizeof(jlong), jdownload, true, arrayLength > 1};

        env->ReleaseLongArrayElements(jarray, arrayPtr, JNI_ABORT);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, kernelArgPtr);

        return obj;
    END_TRY("creating LongArg")
}

jobject Java_LongArg_createOutputInternal (JNIEnv* env, jclass cls, jint jarrayLength){
    BEGIN_TRY
        CHECK_BIGGER(jarrayLength, 0, "illegal array length", NULL)

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{NULL, static_cast<size_t> (jarrayLength) * sizeof(jlong), true, false, true};

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, kernelArgPtr);

        return obj;
    END_TRY("creating LongArg")
}

jlongArray Java_LongArg_asLongArray (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
        auto data = kernelArgJNIPtr->getHostData();
        auto dataSize = kernelArgJNIPtr->kernelArgPtr()->size();

        auto res = env->NewLongArray(dataSize / sizeof(jlong));

        CHECK_NULL(res, NULL)

        env->SetLongArrayRegion(res, 0, dataSize / sizeof(jlong),
                                   reinterpret_cast<const jlong*>(data));
        return res;
    END_TRY("getting LongArg-content")
}

