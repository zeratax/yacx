#include "Kernel.h"
#include "Handle.h"
#include "ProgramArgJNI.hpp"
#include "../../include/cudaexecutor/Logger.hpp"
#include "../../include/cudaexecutor/Exception.hpp"
#include "../../include/cudaexecutor/Kernel.hpp"
#include "../../include/cudaexecutor/ProgramArg.hpp"

using cudaexecutor::loglevel, cudaexecutor::Kernel, cudaexecutor::ProgramArg, jni::ProgramArgJNI, cudaexecutor::CUresultException;

void Java_Kernel_configure(JNIEnv *env, jobject obj, jint jgrid, jint jblock)
{
    try {
        auto kernelPtr = getHandle<Kernel>(env, obj);

        dim3 grid{static_cast<unsigned int> (jgrid)};
        dim3 block{static_cast<unsigned int> (jblock)};

        kernelPtr->configure(grid, block);
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while configuring Kernel");
    }
}

void Java_Kernel_launch(JNIEnv *env, jobject obj, jobjectArray jArgs)
{
    try{
        auto kernelPtr = getHandle<Kernel>(env, obj);

        std::vector<ProgramArg> args;
        args.reserve(env->GetArrayLength(jArgs));
        for(int i = 0; i < args.size(); i++){
            auto jprogramArg = env->GetObjectArrayElement(jArgs, i);
            auto programArgJNIPtr = getHandle<ProgramArgJNI>(env, jprogramArg);
            args.push_back(*programArgJNIPtr->programArgPtr());
        }

        kernelPtr->launch(args);
    }  catch (CUresultException<> err){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, (std::string("Executor failure while launching Kernel: ") + err.what()).c_str());
    } catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while launching Kernel");
    }
}

