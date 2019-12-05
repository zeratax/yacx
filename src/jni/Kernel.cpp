#include "Kernel.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/cudaexecutor/Logger.hpp"
#include "../../include/cudaexecutor/Kernel.hpp"
#include "../../include/cudaexecutor/KernelArg.hpp"
#include "../../include/cudaexecutor/Device.hpp"
#include "../../include/cudaexecutor/KernelTime.hpp"

#include <string>

using cudaexecutor::Kernel, cudaexecutor::KernelArg, cudaexecutor::Device, jni::KernelArgJNI;

void Java_Kernel_configureInternal(JNIEnv *env, jobject obj, jint jgrid1, jint jgrid2, jint jgrid3, jint jblock1, jint jblock2, jint jblock3)
{
    BEGIN_TRY
        auto kernelPtr = getHandle<Kernel>(env, obj);

        dim3 grid{static_cast<unsigned int> (jgrid1), static_cast<unsigned int> (jgrid2), static_cast<unsigned int> (jgrid3)};
        dim3 block{static_cast<unsigned int> (jblock1), static_cast<unsigned int> (jblock2), static_cast<unsigned int> (jblock3)};

        kernelPtr->configure(grid, block);
    END_TRY("configuring Kernel")
}

std::vector<KernelArg> getArguments(JNIEnv* env, jobject jkernel, jobjectArray jArgs)
{
    auto kernelPtr = getHandle<Kernel>(env, jkernel);
    auto argumentsLength = env->GetArrayLength(jArgs);

    std::vector<KernelArg> args;
    args.reserve(argumentsLength);
    for(int i = 0; i < argumentsLength; i++){
        auto jkernelArg = env->GetObjectArrayElement(jArgs, i);
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, jkernelArg);
        args.push_back(*kernelArgJNIPtr->kernelArgPtr());
    }

    return args;
}

jobject createJavaKernelTime(JNIEnv* env, KernelTime* kernelTimePtr){
    jclass cls = env->FindClass("KernelTime");

    if (!cls)
        throw new std::runtime_error("KernelTime class not found!");

    auto methodID = env->GetMethodID(cls, "<init>", "(FFFF)V");
    auto obj = env->NewObject(cls, methodID, kernelTimePtr->upload, kernelTimePtr->download,
            kernelTimePtr->launch, kernelTimePtr->sum);

    return obj;
}

jobject Java_Kernel_launchInternel___3LKernelArg_2(JNIEnv *env, jobject obj, jobjectArray jArgs)
{
    BEGIN_TRY
        auto kernelPtr = getHandle<Kernel>(env, obj);
        auto args = getArguments(env, obj, jArgs);

        auto kernelTimePtr = kernelPtr->launch(args);

        return createJavaKernelTime(env, kernelTimePtr);
    END_TRY("launching Kernel")
}

jobject Java_Kernel_launchInternel___3LKernelArg_2LDevice_2(JNIEnv *env, jobject obj, jobjectArray jArgs, jobject jdevice)
{
    BEGIN_TRY
        auto kernelPtr = getHandle<Kernel>(env, obj);
        auto devicePtr = getHandle<Device>(env, jdevice);

        auto args = getArguments(env, obj, jArgs);

        auto kernelTimePtr = kernelPtr->launch(args, *devicePtr);

        return createJavaKernelTime(env, kernelTimePtr);
    END_TRY("launching Kernel on specific device")
}

jobject Java_Kernel_launchInternel___3LKernelArg_2Ljava_lang_String_2(JNIEnv *env, jobject obj, jobjectArray jArgs, jstring jdevicename)
{
    BEGIN_TRY
        auto kernelPtr = getHandle<Kernel>(env, obj);
        auto devicenamePtr = env->GetStringUTFChars(jdevicename, nullptr);
        std::string devicename{devicenamePtr};

        Device device{devicename};

        env->ReleaseStringUTFChars(jdevicename, devicenamePtr);

        auto args = getArguments(env, obj, jArgs);

        auto kernelTimePtr = kernelPtr->launch(args, device);

        return createJavaKernelTime(env, kernelTimePtr);
    END_TRY("launching Kernel")
}
