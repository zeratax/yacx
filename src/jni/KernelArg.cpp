#include "KernelArg.h"
#include "Handle.h"
#include "ProgramArgJNI.hpp"
#include "../../include/cudaexecutor/Logger.hpp"
#include "../../include/cudaexecutor/Program.hpp"
#include "../../include/cudaexecutor/ProgramArg.hpp"

using cudaexecutor::loglevel, cudaexecutor::Program, cudaexecutor::ProgramArg, jni::ProgramArgJNI;

template <typename T>
jobject createProgramArg(JNIEnv* env, jclass cls, T value)
{
    try {
        ProgramArgJNI* programArgPtr = new ProgramArgJNI{&value, sizeof(T)};

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, programArgPtr);

        return obj;
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating ProgramArg");

        return NULL;
    }
}

jobject Java_KernelArg_create__F(JNIEnv* env, jclass cls, jfloat value){
    return createProgramArg(env, cls, value);
}

jobject Java_KernelArg_create__I(JNIEnv* env, jclass cls, jint value){
    return createProgramArg(env, cls, value);
}

jobject Java_KernelArg_create__D(JNIEnv* env, jclass cls, jdouble value){
    return createProgramArg(env, cls, value);
}

jobject Java_KernelArg_create__Z(JNIEnv* env, jclass cls, jboolean value){
    return createProgramArg(env, cls, value);
}


jobject Java_KernelArg_create___3FZ(JNIEnv* env, jclass cls, jfloatArray jarray, jboolean output){
    try {
        auto arrayPtr = env->GetFloatArrayElements(jarray, nullptr);

        ProgramArgJNI* programArgPtr = new ProgramArgJNI{arrayPtr, env->GetArrayLength(jarray) * sizeof(jfloat), output, true, true};

        env->ReleaseFloatArrayElements(jarray, arrayPtr, JNI_ABORT);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, programArgPtr);

        return obj;
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating ProgramArg");

        return NULL;
    }
}

jobject Java_KernelArg_create___3IZ(JNIEnv* env, jclass cls, jintArray jarray, jboolean output){
    try {
        auto arrayPtr = env->GetIntArrayElements(jarray, nullptr);

        ProgramArgJNI* programArgPtr = new ProgramArgJNI{arrayPtr, env->GetArrayLength(jarray) * sizeof(jfloat), output, true, true};

        env->ReleaseIntArrayElements(jarray, arrayPtr, JNI_ABORT);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, programArgPtr);

        return obj;
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating ProgramArg");

        return NULL;
    }
}

jobject Java_KernelArg_create___3DZ(JNIEnv* env, jclass cls, jdoubleArray jarray, jboolean output){
    try {
        auto arrayPtr = env->GetDoubleArrayElements(jarray, nullptr);

        ProgramArgJNI* programArgPtr = new ProgramArgJNI{arrayPtr, env->GetArrayLength(jarray) * sizeof(jfloat), output, true, true};

        env->ReleaseDoubleArrayElements(jarray, arrayPtr, JNI_ABORT);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, programArgPtr);

        return obj;
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating ProgramArg");

        return NULL;
    }
}

jobject Java_KernelArg_create___3ZZ(JNIEnv* env, jclass cls, jbooleanArray jarray, jboolean output){
    try {
        auto arrayPtr = env->GetBooleanArrayElements(jarray, nullptr);

        ProgramArgJNI* programArgPtr = new ProgramArgJNI{arrayPtr, env->GetArrayLength(jarray) * sizeof(jfloat), output, true, true};

        env->ReleaseBooleanArrayElements(jarray, arrayPtr, JNI_ABORT);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, programArgPtr);

        return obj;
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating ProgramArg");

        return NULL;
    }
}


jobject Java_KernelArg_createOutput(JNIEnv* env, jclass cls, jlong argSize){
    try {
        ProgramArgJNI* programArgPtr = new ProgramArgJNI{NULL, static_cast<size_t> (argSize), true, false, true};

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, programArgPtr);

        return obj;
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating ProgramArg");

        return NULL;
    }
}


jfloatArray Java_KernelArg_asFloatArray(JNIEnv* env, jobject obj){
    try {
        auto programArgJNIPtr = getHandle<ProgramArgJNI>(env, obj);
        auto data = programArgJNIPtr->programArgPtr()->content();
        auto dataSize = programArgJNIPtr->programArgPtr()->size();

        auto res = env->NewFloatArray(dataSize / sizeof(jfloat));
        if (res == nullptr) return nullptr;

        env->SetFloatArrayRegion(res, 0, dataSize / sizeof(jfloat),
                                 reinterpret_cast<const jfloat*>(data));
        return res;
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating ProgramArg");

        return NULL;
    }
}

jintArray Java_KernelArg_asIntArray(JNIEnv* env, jobject obj){
    try {
        auto programArgJNIPtr = getHandle<ProgramArgJNI>(env, obj);
        auto data = programArgJNIPtr->programArgPtr()->content();
        auto dataSize = programArgJNIPtr->programArgPtr()->size();

        auto res = env->NewIntArray(dataSize / sizeof(jint));
        if (res == nullptr) return nullptr;

        env->SetIntArrayRegion(res, 0, dataSize / sizeof(jint),
                                 reinterpret_cast<const jint*>(data));
        return res;
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating ProgramArg");

        return NULL;
    }
}

jdoubleArray Java_KernelArg_asDoubleArray(JNIEnv* env, jobject obj){
    try {
        auto programArgJNIPtr = getHandle<ProgramArgJNI>(env, obj);
        auto data = programArgJNIPtr->programArgPtr()->content();
        auto dataSize = programArgJNIPtr->programArgPtr()->size();

        auto res = env->NewDoubleArray(dataSize / sizeof(jdouble));
        if (res == nullptr) return nullptr;

        env->SetDoubleArrayRegion(res, 0, dataSize / sizeof(jdouble),
                                 reinterpret_cast<const jdouble*>(data));
        return res;
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating ProgramArg");

        return NULL;
    }
}

jbooleanArray Java_KernelArg_asBooleanArray(JNIEnv* env, jobject obj){
    try {
        auto programArgJNIPtr = getHandle<ProgramArgJNI>(env, obj);
        auto data = programArgJNIPtr->programArgPtr()->content();
        auto dataSize = programArgJNIPtr->programArgPtr()->size();

        auto res = env->NewBooleanArray(dataSize / sizeof(jboolean));
        if (res == nullptr) return nullptr;

        env->SetBooleanArrayRegion(res, 0, dataSize / sizeof(jboolean),
                                 reinterpret_cast<const jboolean*>(data));
        return res;
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating ProgramArg");

        return NULL;
    }
}