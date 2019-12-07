#include "Kernel.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/yacx/Logger.hpp"
#include "../../include/yacx/Kernel.hpp"
#include "../../include/yacx/KernelArgs.hpp"
#include "../../include/yacx/Device.hpp"
#include "../../include/yacx/KernelTime.hpp"

#include <string>

using yacx::Kernel, yacx::KernelArg, yacx::KernelArgs, yacx::KernelTime, yacx::Device, jni::KernelArgJNI;

void Java_Kernel_configureInternal(JNIEnv *env, jobject obj, jint jgrid0, jint jgrid1, jint jgrid2, jint jblock0, jint jblock1, jint jblock2)
{
    BEGIN_TRY
        CHECK_BIGGER(jgrid0, 0, "illegal size for grid0")
        CHECK_BIGGER(jgrid1, 0, "illegal size for grid1")
        CHECK_BIGGER(jgrid2, 0, "illegal size for grid2")
        CHECK_BIGGER(jblock0, 0, "illegal size for block0")
        CHECK_BIGGER(jblock1, 0, "illegal size for block1")
        CHECK_BIGGER(jblock2, 0, "illegal size for block2")

        auto kernelPtr = getHandle<Kernel>(env, obj);

        dim3 grid{static_cast<unsigned int> (jgrid0), static_cast<unsigned int> (jgrid1), static_cast<unsigned int> (jgrid2)};
        dim3 block{static_cast<unsigned int> (jblock0), static_cast<unsigned int> (jblock1), static_cast<unsigned int> (jblock2)};

        kernelPtr->configure(grid, block);
    END_TRY("configuring Kernel")
}

std::vector<KernelArg> getArguments(JNIEnv* env, jobject jkernel, jobjectArray jArgs)
{
    CHECK_NULL(jArgs)

    auto kernelPtr = getHandle<Kernel>(env, jkernel);
    auto argumentsLength = env->GetArrayLength(jArgs);

    CHECK_BIGGER(argumentsLength, 0, "illegal array length")

    std::vector<KernelArg> args;
    args.reserve(argumentsLength);
    for(int i = 0; i < argumentsLength; i++){
        auto jkernelArg = env->GetObjectArrayElement(jArgs, i);

        CHECK_NULL(jkernelArg)

        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, jkernelArg);
        args.push_back(*kernelArgJNIPtr->kernelArgPtr());
    }

    return args;
}

jobject createJavaKernelTime(JNIEnv* env, KernelTime* kernelTimePtr){
    jclass cls = getClass(env, "KernelTime");

    auto methodID = env->GetMethodID(cls, "<init>", "(FFFF)V");
    auto obj = env->NewObject(cls, methodID, kernelTimePtr->upload, kernelTimePtr->download,
            kernelTimePtr->launch, kernelTimePtr->sum);

    return obj;
}

jobject Java_Kernel_launchInternal___3LKernelArg_2(JNIEnv *env, jobject obj, jobjectArray jArgs)
{
    BEGIN_TRY
        auto kernelPtr = getHandle<Kernel>(env, obj);
        auto args = getArguments(env, obj, jArgs);

        auto kernelTimePtr = kernelPtr->launch(KernelArgs{args});

        return createJavaKernelTime(env, &kernelTimePtr);
    END_TRY("launching Kernel")
}

jobject Java_Kernel_launchInternal___3LKernelArg_2LDevice_2(JNIEnv *env, jobject obj, jobjectArray jArgs, jobject jdevice)
{
    BEGIN_TRY
        auto kernelPtr = getHandle<Kernel>(env, obj);
        auto devicePtr = getHandle<Device>(env, jdevice);

        auto args = getArguments(env, obj, jArgs);

        auto kernelTimePtr = kernelPtr->launch(KernelArgs{args}, *devicePtr);

        return createJavaKernelTime(env, &kernelTimePtr);
    END_TRY("launching Kernel on specific device")
}

jobject Java_Kernel_launchInternal___3LKernelArg_2Ljava_lang_String_2(JNIEnv *env, jobject obj, jobjectArray jArgs, jstring jdevicename)
{
    BEGIN_TRY
        CHECK_NULL(jdevicename)

        auto kernelPtr = getHandle<Kernel>(env, obj);
        auto devicenamePtr = env->GetStringUTFChars(jdevicename, nullptr);
        std::string devicename{devicenamePtr};

        Device device{devicename};

        env->ReleaseStringUTFChars(jdevicename, devicenamePtr);

        auto args = getArguments(env, obj, jArgs);

        auto kernelTimePtr = kernelPtr->launch(KernelArgs{args}, device);

        return createJavaKernelTime(env, &kernelTimePtr);
    END_TRY("launching Kernel")
}
