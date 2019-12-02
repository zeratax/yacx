#include "Kernel.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/cudaexecutor/Logger.hpp"
#include "../../include/cudaexecutor/Kernel.hpp"
#include "../../include/cudaexecutor/KernelArg.hpp"

using cudaexecutor::loglevel, cudaexecutor::Kernel, cudaexecutor::KernelArg, jni::KernelArgJNI;

void Java_Kernel_configureInternal(JNIEnv *env, jobject obj, jint jgrid1, jint jgrid2, jint jgrid3, jint jblock1, jint jblock2, jint jblock3)
{
    BEGIN_TRY
        auto kernelPtr = getHandle<Kernel>(env, obj);

        dim3 grid{static_cast<unsigned int> (jgrid1), static_cast<unsigned int> (jgrid2), static_cast<unsigned int> (jgrid3)};
        dim3 block{static_cast<unsigned int> (jblock1), static_cast<unsigned int> (jblock2), static_cast<unsigned int> (jblock3)};

        kernelPtr->configure(grid, block);
    END_TRY("configuring Kernel")
}

void Java_Kernel_launchInternel(JNIEnv *env, jobject obj, jobjectArray jArgs)
{
    BEGIN_TRY
        auto kernelPtr = getHandle<Kernel>(env, obj);
        auto argumentsLength = env->GetArrayLength(jArgs);

        std::vector<KernelArg> args;
        args.reserve(argumentsLength);
        for(int i = 0; i < argumentsLength; i++){
            auto jkernelArg = env->GetObjectArrayElement(jArgs, i);
            auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, jkernelArg);
            args.push_back(*kernelArgJNIPtr->kernelArgPtr());
        }

        kernelPtr->launch(args);
    END_TRY("launching Kernel")
}

