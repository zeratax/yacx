#include "ValueArg.h"
#include "Handle.h"
#include "ProgramArgJNI.hpp"
#include "../../include/cudaexecutor/Logger.hpp"
#include "../../include/cudaexecutor/Program.hpp"
#include "../../include/cudaexecutor/ProgramArg.hpp"

using cudaexecutor::loglevel, cudaexecutor::Program, cudaexecutor::ProgramArg, jni::ProgramArgJNI;

template <typename T>
jobject createProgramArg (JNIEnv* env, jclass cls, T value){
    BEGIN_TRY
        ProgramArgJNI* programArgPtr = new ProgramArgJNI{&value, sizeof(T)};

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, programArgPtr);

        return obj;
    END_TRY("creating ProgramArg")
}

jobject Java_ValueArg_createInternal__F (JNIEnv* env, jclass cls, jfloat value){
    return createProgramArg(env, cls, value);
}

jobject Java_ValueArg_createInternal__I (JNIEnv* env, jclass cls, jint value){
    return createProgramArg(env, cls, value);
}

jobject Java_ValueArg_createInternal__J (JNIEnv* env, jclass cls, jlong value){
    return createProgramArg(env, cls, value);
}

jobject Java_ValueArg_createInternal__D (JNIEnv* env, jclass cls, jdouble value){
    return createProgramArg(env, cls, value);
}

jobject Java_ValueArg_createInternal__Z (JNIEnv* env, jclass cls, jboolean value){
    return createProgramArg(env, cls, value);
}

jfloat Java_ValueArg_asFloatInternal (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto programArgJNIPtr = getHandle<ProgramArgJNI>(env, obj);
        return *static_cast<float*> (programArgJNIPtr->getHostData());
    END_TRY("getting float-value from KernelArg")
}

jint Java_ValueArg_asIntInternal (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto programArgJNIPtr = getHandle<ProgramArgJNI>(env, obj);
        return *static_cast<int*> (programArgJNIPtr->getHostData());
    END_TRY("getting int-value from KernelArg")
}

jlong Java_ValueArg_asLongInternal (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto programArgJNIPtr = getHandle<ProgramArgJNI>(env, obj);
        return *static_cast<long*> (programArgJNIPtr->getHostData());
    END_TRY("getting long-value from KernelArg")
}

jdouble Java_ValueArg_asDoubleInternal (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto programArgJNIPtr = getHandle<ProgramArgJNI>(env, obj);
        return *static_cast<double*> (programArgJNIPtr->getHostData());
    END_TRY("getting double-value from KernelArg")
}

jboolean Java_ValueArg_asBooleanInternal (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto programArgJNIPtr = getHandle<ProgramArgJNI>(env, obj);
        return *static_cast<bool*> (programArgJNIPtr->getHostData());
    END_TRY("getting boolean-value from KernelArg")
}

