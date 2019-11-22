#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#include <jni.h>
#pragma GCC diagnostic pop

#include "Handle.h"
#include "ProgramArg.hpp"
#include "../../include/cudaexecutor/Kernel.hpp"
#include "../../include/cudaexecutor/Exception.hpp"
#include "../../include/cudaexecutor/Logger.hpp"

void Java_Kernel_compile(JNIEnv *env, jobject obj)
{
    auto ptr = getHandle(env, obj);

    try {
        ptr->compile();
    }catch(nvrtcResultException err){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(logLevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, (std::string("Executor failure: ") + err.what());
    }catch(...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure");
    }


}

void Java_Kernel_configure(JNIEnv *env, jobject obj, jint grid, jint block)
{
    auto ptr = getHandle(env, obj);
    dim3 gridS(grid);
    dim3 blockS(block);

    ptr->configure(grid, block);
}

void Java_Kernel_launch(JNIEnv *env, jobject jKernel, jobjectArray jArgs)
{
    try{
        auto ptr = getHandle(env, obj);

        std::vector<KernelArg*> args(env->GetArrayLength(jArgs));
        int i = 0;
        for(auto& p : args){
            auto obj = env->GetObjectArrayElement(jArgs, i);
            p = getHandle(env, obj);
            ++i;
        }

        ptr->launch();
    } catch(nvrtcResultException err){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(logLevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, (std::string("Executor failure: ") + err.what());
    }catch(...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure");
    }
}

