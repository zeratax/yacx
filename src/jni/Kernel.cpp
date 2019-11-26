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
    BEGIN_TRY
        auto kernelPtr = getHandle<Kernel>(env, obj);

        dim3 grid{static_cast<unsigned int> (jgrid)};
        dim3 block{static_cast<unsigned int> (jblock)};

        kernelPtr->configure(grid, block);
    END_TRY("configuring Kernel")
}

void Java_Kernel_launch(JNIEnv *env, jobject obj, jobjectArray jArgs)
{
    BEGIN_TRY
        auto kernelPtr = getHandle<Kernel>(env, obj);
        auto argumentsLength = env->GetArrayLength(jArgs);

        std::vector<ProgramArg> args;
        args.reserve(argumentsLength);
        for(int i = 0; i < argumentsLength; i++){
            auto jprogramArg = env->GetObjectArrayElement(jArgs, i);
            auto programArgJNIPtr = getHandle<ProgramArgJNI>(env, jprogramArg);
            args.push_back(*programArgJNIPtr->programArgPtr());
            logger(loglevel::ERROR) << "SIZE Argument " << i << " " << programArgJNIPtr->programArgPtr()->size();
        }

        kernelPtr->launch(args);
    END_TRY("launching Kernel")
}

