#include "KernelArg.h"
#include "../../include/cudaexecutor/ProgramArg.hpp"

template <typename T>
jobject createProgrammArg(JNIEnv* env, jclass cls, T value)
{
    try {
        auto ptr = new ProgramArg{value};

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, ptr);

        return obj;
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating ProgramArg");

        return NULL;
    }
}

JNIEXPORT jobject JNICALL Java_KernelArg_create__F(JNIEnv* env, jclass cls, jfloat value){
    return createProgrammArg(env, cls, value);
}

JNIEXPORT jobject JNICALL Java_KernelArg_create__I(JNIEnv* env, jclass cls, jint, value){
    return createProgrammArg(env, cls, value);
}

JNIEXPORT jobject JNICALL Java_KernelArg_create__D(JNIEnv* env, jclass cls, jdouble value){
    return createProgrammArg(env, cls, value);
}

JNIEXPORT jobject JNICALL Java_KernelArg_create__Z(JNIEnv* env, jclass cls, jboolean, value){
    return createProgrammArg(env, cls, value);
}


JNIEXPORT jobject JNICALL Java_KernelArg_create___3FZ(JNIEnv* env, jclass cls, jfloatArray jarray, jboolean output){
    try {
        auto arrayPtr = env->GetFloatArrayElements(jarray, nullptr);

        auto ptr = new Program{arrayPtr, env->GetArrayLength(data) * sizeof(jfloat), output, true, true};

        env->ReleaseFloatArrayElements(jarray, arrayPtr, JNI_ABORT);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, ptr);

        return obj;
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating ProgramArg");

        return NULL;
    }
}

JNIEXPORT jobject JNICALL Java_KernelArg_create___3IZ(JNIEnv* env, jclass cls, jintArray jarray, jboolean output){
    try {
        auto arrayPtr = env->GetIntArrayElements(jarray, nullptr);

        auto ptr = new Program{arrayPtr, env->GetArrayLength(data) * sizeof(jfloat), output, true, true};

        env->ReleaseIntArrayElements(jarray, arrayPtr, JNI_ABORT);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, ptr);

        return obj;
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating ProgramArg");

        return NULL;
    }
}

JNIEXPORT jobject JNICALL Java_KernelArg_create___3DZ(JNIEnv* env, jclass cls, jdoubleArray jarray, jboolean output){
    try {
        auto arrayPtr = env->GetDoubleArrayElements(jarray, nullptr);

        auto ptr = new Program{arrayPtr, env->GetArrayLength(data) * sizeof(jfloat), output, true, true};

        env->ReleaseDoubleArrayElements(jarray, arrayPtr, JNI_ABORT);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, ptr);

        return obj;
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating ProgramArg");

        return NULL;
    }
}

JNIEXPORT jobject JNICALL Java_KernelArg_create___3ZZ(JNIEnv* env, jclass cls, jbooleanArray jarray, jboolean output){
    try {
        auto arrayPtr = env->GetBooleanArrayElements(jarray, nullptr);

        auto ptr = new Program{arrayPtr, env->GetArrayLength(data) * sizeof(jfloat), output, true, true};

        env->ReleaseBooleanArrayElements(jarray, arrayPtr, JNI_ABORT);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, ptr);

        return obj;
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating ProgramArg");

        return NULL;
    }
}


JNIEXPORT jobject JNICALL Java_KernelArg_createOutput(JNIEnv* env, jclass cls, jlong argSize){
    try {
        auto arrayPtr = malloc(argSize);

        auto ptr = new ProgramArg{arrayPtr, argSize, true, false, true};

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, ptr);

        return obj;
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating ProgramArg");

        return NULL;
    }
}


JNIEXPORT jfloatArray JNICALL Java_KernelArg_asFloatArray(JNIEnv* env, jobject obj){
    try {
        auto ptr = getHandle<ProgramArg>(env, obj);
        auto& vec = ptr->data();

        auto res = env->NewFloatArray(vec.size() / sizeof(jfloat));
        if (res == nullptr) return nullptr;

        env->SetFloatArrayRegion(res, 0, vec.size() / sizeof(jfloat),
                                 reinterpret_cast<jfloat*>(vec.hostBuffer().data()));
        return res;
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating ProgramArg");

        return NULL;
    }
}

JNIEXPORT jintArray JNICALL Java_KernelArg_asIntArray(JNIEnv* env, jobject obj){
    try {
        auto ptr = getHandle<ProgramArg>(env, obj);
        auto& vec = ptr->data();

        auto res = env->NewIntArray(vec.size() / sizeof(jint));
        if (res == nullptr) return nullptr;

        env->SetIntArrayRegion(res, 0, vec.size() / sizeof(jint),
                                 reinterpret_cast<jint*>(vec.hostBuffer().data()));
        return res;
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating ProgramArg");

        return NULL;
    }
}

JNIEXPORT jdoubleArray JNICALL Java_KernelArg_asDoubleArray(JNIEnv* env, jobject obj){
    try {
        auto ptr = getHandle<ProgramArg>(env, obj);
        auto& vec = ptr->data();

        auto res = env->NewDoubleArray(vec.size() / sizeof(jdouble));
        if (res == nullptr) return nullptr;

        env->SetDoubleArrayRegion(res, 0, vec.size() / sizeof(jdouble),
                                 reinterpret_cast<jdouble*>(vec.hostBuffer().data()));
        return res;
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating ProgramArg");

        return NULL;
    }
}

JNIEXPORT jbooleanArray JNICALL Java_KernelArg_asBooleanArray(JNIEnv* env, jobject obj){
    try {
        auto ptr = getHandle<ProgramArg>(env, obj);
        auto& vec = ptr->data();

        auto res = env->NewBooleanArray(vec.size() / sizeof(jboolean));
        if (res == nullptr) return nullptr;

        env->SetBooleanArrayRegion(res, 0, vec.size() / sizeof(jboolean),
                                 reinterpret_cast<jboolean*>(vec.hostBuffer().data()));
        return res;
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating ProgramArg");

        return NULL;
    }
}