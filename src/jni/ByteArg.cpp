#include "ByteArg.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/yacx/KernelArgs.hpp"

using yacx::KernelArg, jni::KernelArgJNI;

jobject Java_ByteArg_createInternal (JNIEnv* env, jclass cls, jbyteArray jarray, jboolean jdownload){
    BEGIN_TRY
        CHECK_NULL(jarray, NULL)

        auto arrayPtr = env->GetByteArrayElements(jarray, NULL);
        auto arrayLength = env->GetArrayLength(jarray);

        CHECK_BIGGER(arrayLength, 0, "illegal array length", NULL)

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{arrayPtr, arrayLength * sizeof(jbyte), jdownload, true, arrayLength > 1};

        env->ReleaseByteArrayElements(jarray, arrayPtr, JNI_ABORT);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, kernelArgPtr);

        return obj;
    END_TRY("creating ByteArg")
}

jobject Java_ByteArg_createOutputInternal (JNIEnv* env, jclass cls, jint jarrayLength){
    BEGIN_TRY
        CHECK_BIGGER(jarrayLength, 0, "illegal array length", NULL)

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{NULL, static_cast<size_t> (jarrayLength) * sizeof(jbyte), true, false, true};

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, kernelArgPtr);

        return obj;
    END_TRY("creating ByteArg")
}

jbyteArray Java_ByteArg_asByteArray (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
        auto data = kernelArgJNIPtr->getHostData();
        auto dataSize = kernelArgJNIPtr->kernelArgPtr()->size();

        auto res = env->NewByteArray(dataSize / sizeof(jbyte));

        CHECK_NULL(res, NULL)

        env->SetByteArrayRegion(res, 0, dataSize / sizeof(jbyte),
                                   reinterpret_cast<const jbyte*>(data));
        return res;
    END_TRY("getting ByteArg-content")
}

