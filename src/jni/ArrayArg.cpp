#include "ArrayArg.h"
#include "Handle.h"
#include "ProgramArgJNI.hpp"
#include "../../include/cudaexecutor/Logger.hpp"
#include "../../include/cudaexecutor/Program.hpp"
#include "../../include/cudaexecutor/ProgramArg.hpp"

using cudaexecutor::loglevel, cudaexecutor::Program, cudaexecutor::ProgramArg, jni::ProgramArgJNI;

jobject Java_ArrayArg_createInternal___3FZ(JNIEnv* env, jclass cls, jfloatArray jarray, jboolean output){
    BEGIN_TRY
        auto arrayPtr = env->GetFloatArrayElements(jarray, nullptr);

        ProgramArgJNI* programArgPtr = new ProgramArgJNI{arrayPtr, env->GetArrayLength(jarray) * sizeof(jfloat), output, true, true};

        env->ReleaseFloatArrayElements(jarray, arrayPtr, JNI_ABORT);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, programArgPtr);

        return obj;
    END_TRY("creating ProgramArg")
}

jobject Java_ArrayArg_createInternal___3IZ(JNIEnv* env, jclass cls, jintArray jarray, jboolean output){
    BEGIN_TRY
        auto arrayPtr = env->GetIntArrayElements(jarray, nullptr);

        ProgramArgJNI* programArgPtr = new ProgramArgJNI{arrayPtr, env->GetArrayLength(jarray) * sizeof(jint), output, true, true};

        env->ReleaseIntArrayElements(jarray, arrayPtr, JNI_ABORT);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, programArgPtr);

        return obj;
    END_TRY("creating ProgramArg")
}

jobject Java_ArrayArg_createInternal___3JZ(JNIEnv* env, jclass cls, jlongArray jarray, jboolean output){
    BEGIN_TRY
        auto arrayPtr = env->GetLongArrayElements(jarray, nullptr);

        ProgramArgJNI* programArgPtr = new ProgramArgJNI{arrayPtr, env->GetArrayLength(jarray) * sizeof(jlong), output, true, true};

        env->ReleaseLongArrayElements(jarray, arrayPtr, JNI_ABORT);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, programArgPtr);

        return obj;
    END_TRY("creating ProgramArg")
}

jobject Java_ArrayArg_createInternal___3DZ(JNIEnv* env, jclass cls, jdoubleArray jarray, jboolean output){
    BEGIN_TRY
        auto arrayPtr = env->GetDoubleArrayElements(jarray, nullptr);

        ProgramArgJNI* programArgPtr = new ProgramArgJNI{arrayPtr, env->GetArrayLength(jarray) * sizeof(jdouble), output, true, true};

        env->ReleaseDoubleArrayElements(jarray, arrayPtr, JNI_ABORT);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, programArgPtr);

        return obj;
    END_TRY("creating ProgramArg")
}

jobject Java_ArrayArg_createInternal___3ZZ(JNIEnv* env, jclass cls, jbooleanArray jarray, jboolean output){
    BEGIN_TRY
        auto arrayPtr = env->GetBooleanArrayElements(jarray, nullptr);

        ProgramArgJNI* programArgPtr = new ProgramArgJNI{arrayPtr, env->GetArrayLength(jarray) * sizeof(jboolean), output, true, true};

        env->ReleaseBooleanArrayElements(jarray, arrayPtr, JNI_ABORT);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, programArgPtr);

        return obj;
    END_TRY("creating ProgramArg")
}


jobject Java_ArrayArg_createOutputInternal(JNIEnv* env, jclass cls, jlong argSize){
    BEGIN_TRY
        ProgramArgJNI* programArgPtr = new ProgramArgJNI{NULL, static_cast<size_t> (argSize), true, false, true};

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, programArgPtr);

        return obj;
    END_TRY("creating ProgramArg")
}


jfloatArray Java_ArrayArg_asFloatArrayInternal(JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto programArgJNIPtr = getHandle<ProgramArgJNI>(env, obj);
        auto data = programArgJNIPtr->getHostData();
        auto dataSize = programArgJNIPtr->programArgPtr()->size();

        auto res = env->NewFloatArray(dataSize / sizeof(jfloat));
        if (res == nullptr) return nullptr;

        env->SetFloatArrayRegion(res, 0, dataSize / sizeof(jfloat),
                                 reinterpret_cast<const jfloat*>(data));
        return res;
    END_TRY("converting ProgramArg to Java-FloatArray")
}

jintArray Java_ArrayArg_asIntArrayInternal(JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto programArgJNIPtr = getHandle<ProgramArgJNI>(env, obj);
        auto data = programArgJNIPtr->programArgPtr()->content();
        auto dataSize = programArgJNIPtr->programArgPtr()->size();

        auto res = env->NewIntArray(dataSize / sizeof(jint));
        if (res == nullptr) return nullptr;

        env->SetIntArrayRegion(res, 0, dataSize / sizeof(jint),
                               reinterpret_cast<const jint*>(data));
        return res;
    END_TRY("converting ProgramArg to Java-IntArray")
}

jlongArray Java_ArrayArg_asLongArrayInternal(JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto programArgJNIPtr = getHandle<ProgramArgJNI>(env, obj);
        auto data = programArgJNIPtr->programArgPtr()->content();
        auto dataSize = programArgJNIPtr->programArgPtr()->size();

        auto res = env->NewLongArray(dataSize / sizeof(jlong));
        if (res == nullptr) return nullptr;

        env->SetLongArrayRegion(res, 0, dataSize / sizeof(jlong),
                               reinterpret_cast<const jlong*>(data));
        return res;
    END_TRY("converting ProgramArg to Java-IntArray")
}

jdoubleArray Java_ArrayArg_asDoubleArrayInternal(JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto programArgJNIPtr = getHandle<ProgramArgJNI>(env, obj);
        auto data = programArgJNIPtr->programArgPtr()->content();
        auto dataSize = programArgJNIPtr->programArgPtr()->size();

        auto res = env->NewDoubleArray(dataSize / sizeof(jdouble));
        if (res == nullptr) return nullptr;

        env->SetDoubleArrayRegion(res, 0, dataSize / sizeof(jdouble),
                                  reinterpret_cast<const jdouble*>(data));
        return res;
    END_TRY("converting ProgramArg to Java-DoubleArray")
}

jbooleanArray Java_ArrayArg_asBooleanArrayInternal(JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto programArgJNIPtr = getHandle<ProgramArgJNI>(env, obj);
        auto data = programArgJNIPtr->programArgPtr()->content();
        auto dataSize = programArgJNIPtr->programArgPtr()->size();

        auto res = env->NewBooleanArray(dataSize / sizeof(jboolean));
        if (res == nullptr) return nullptr;

        env->SetBooleanArrayRegion(res, 0, dataSize / sizeof(jboolean),
                                   reinterpret_cast<const jboolean*>(data));
        return res;
    END_TRY("converting ProgramArg to Java-BooleanArray")
}