#include "ValueArg.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/cudaexecutor/Logger.hpp"
#include "../../include/cudaexecutor/Program.hpp"
#include "../../include/cudaexecutor/KernelArg.hpp"

using cudaexecutor::Program, cudaexecutor::KernelArg, jni::KernelArgJNI;

template <typename T>
jobject createKernelArg (JNIEnv* env, jclass cls, T value){
    BEGIN_TRY
        KernelArgJNI* kernelArgPtr = new KernelArgJNI{&value, sizeof(T)};

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, kernelArgPtr);

        return obj;
    END_TRY("creating KernelArg")
}

jobject Java_ValueArg_createInternal__F (JNIEnv* env, jclass cls, jfloat value){
    return createKernelArg(env, cls, value);
}

jobject Java_ValueArg_createInternal__I (JNIEnv* env, jclass cls, jint value){
    return createKernelArg(env, cls, value);
}

jobject Java_ValueArg_createInternal__J (JNIEnv* env, jclass cls, jlong value){
    return createKernelArg(env, cls, value);
}

jobject Java_ValueArg_createInternal__D (JNIEnv* env, jclass cls, jdouble value){
    return createKernelArg(env, cls, value);
}

jobject Java_ValueArg_createInternal__Z (JNIEnv* env, jclass cls, jboolean value){
    return createKernelArg(env, cls, value);
}

jfloat Java_ValueArg_asFloatInternal (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
        return *static_cast<float*> (kernelArgJNIPtr->getHostData());
    END_TRY("getting float-value from KernelArg")
}

jint Java_ValueArg_asIntInternal (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
        return *static_cast<int*> (kernelArgJNIPtr->getHostData());
    END_TRY("getting int-value from KernelArg")
}

jlong Java_ValueArg_asLongInternal (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
        return *static_cast<long*> (kernelArgJNIPtr->getHostData());
    END_TRY("getting long-value from KernelArg")
}

jdouble Java_ValueArg_asDoubleInternal (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
        return *static_cast<double*> (kernelArgJNIPtr->getHostData());
    END_TRY("getting double-value from KernelArg")
}

jboolean Java_ValueArg_asBooleanInternal (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
        return *static_cast<bool*> (kernelArgJNIPtr->getHostData());
    END_TRY("getting boolean-value from KernelArg")
}

