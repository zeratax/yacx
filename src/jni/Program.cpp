#include "Program.h"
#include "Handle.h"
#include "../../include/cudaexecutor/Logger.hpp"
#include "../../include/cudaexecutor/Source.hpp"
#include "../../include/cudaexecutor/Program.hpp"
#include "../../include/cudaexecutor/Kernel.hpp"
#include "../../include/cudaexecutor/Exception.hpp"

using cudaexecutor::loglevel, cudaexecutor::Source, cudaexecutor::Program, cudaexecutor::Kernel, cudaexecutor::nvrtcResultException;

jobject Java_Program_create (JNIEnv* env, jclass cls, jstring jkernelSource, jstring jkernelName){
    try {
        auto kernelSourcePtr = env->GetStringUTFChars(jkernelSource, nullptr);
        auto kernelNamePtr = env->GetStringUTFChars(jkernelName, nullptr);

        Source source{kernelSourcePtr};
        Program* programPtr = new Program{source.program(kernelNamePtr)};

        env->ReleaseStringUTFChars(jkernelSource, kernelSourcePtr);
        env->ReleaseStringUTFChars(jkernelName, kernelNamePtr);

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, programPtr);

        return obj;
    } catch (nvrtcResultException<> err){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, (std::string("Executor failure while creating Program: ") + err.what()).c_str());
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while creating Program");

        return NULL;
    }
}

jobject Java_Program_compile (JNIEnv* env, jobject obj){
    try{
        auto programPtr = getHandle<Program>(env, obj);

        Kernel* kernelPtr = new Kernel{programPtr->compile()};

        jclass jKernel = env->FindClass("Kernel");
        auto methodID = env->GetMethodID(jKernel, "<init>", "(J)V");
        auto kernelObj = env->NewObject(jKernel, methodID, kernelPtr);

        return kernelObj;
    } catch (nvrtcResultException<> err){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, (std::string("Executor failure while creating Program: ") + err.what()).c_str());
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while compiling Kernel");

        return NULL;
    }
}