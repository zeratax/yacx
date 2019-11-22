#include <Program.h>
#include <../../include/cudaexecutor/Program.hpp>
#include <../../include/cudaexecutor/Logger.hpp>
#include <../../include/cudaexecutor/Exception.hpp>

using cudaexecutor::Program;

JNIEXPORT jobject JNICALL Java_Program_create (JNIEnv* env, jclass cls, jstring jkernelString){
    try {
        auto kernelStringPtr = env->GetStringUTFChars(jkernelString, nullptr);

        auto ptr = new Program{*kernelStringPtr};

        env->ReleaseStringUTFChars(jkernelString, kernelStringPtr);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, ptr);

        return obj;
    } catch (CUresultException error){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, (std::string("Executor failure while creating Program: ") + err.what());
    }catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating Program");
    }
}

JNIEXPORT jobject JNICALL Java_Program_kernel (JNIEnv* env, jobject obj, jstring jkernelName){
    try{
        auto kernelNamePtr = env->GetStringUTFChars(jkernelName, nullptr);

        auto ptr = getHandle<executor::Kernel>(env, obj);
        auto kernelPtr = ptr->kernel(kernelNamePtr);

        env->ReleaseStringUTFChars(jkernelName, kernelNamePtr);

        jclass jClass = env->FindClass("Kernel");
        auto methodID = env->GetMethodID(jClass, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, kernelPtr);

        return obj;
    } catch (CUresultException error){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, (std::string("Executor failure while creating Kernel: ") + err.what());
    }catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating Kernel");
    }
}