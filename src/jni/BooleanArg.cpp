#include "BooleanArg.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/cudaexecutor/KernelArgs.hpp"

using cudaexecutor::KernelArg, jni::KernelArgJNI;

jobject Java_BooleanArg_createInternal (JNIEnv* env, jclass cls, jbooleanArray jarray, jboolean jdownload){
    BEGIN_TRY
        CHECK_NULL(jarray, NULL)

        auto arrayPtr = env->GetBooleanArrayElements(jarray, NULL);
        auto arrayLength = env->GetArrayLength(jarray);

        CHECK_BIGGER(arrayLength, 0, "illegal array length", NULL)

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{arrayPtr, arrayLength * sizeof(jboolean), jdownload, true, arrayLength > 1};

        env->ReleaseBooleanArrayElements(jarray, arrayPtr, JNI_ABORT);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, kernelArgPtr);

        return obj;
    END_TRY("creating BooleanArg")
}

jobject Java_BooleanArg_createOutputInternal (JNIEnv* env, jclass cls, jint jarrayLength){
    BEGIN_TRY
        CHECK_BIGGER(jarrayLength, 0, "illegal array length", NULL)

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{NULL, static_cast<size_t> (jarrayLength) * sizeof(jboolean), true, false, true};

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, kernelArgPtr);

        return obj;
    END_TRY("creating BooleanArg")
}

jbooleanArray Java_BooleanArg_asBooleanArray (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
        auto data = kernelArgJNIPtr->getHostData();
        auto dataSize = kernelArgJNIPtr->kernelArgPtr()->size();

        auto res = env->NewBooleanArray(dataSize / sizeof(jboolean));

        CHECK_NULL(res, NULL)

        env->SetBooleanArrayRegion(res, 0, dataSize / sizeof(jboolean),
                                   reinterpret_cast<const jboolean*>(data));
        return res;
    END_TRY("getting BooleanArg-content")
}

