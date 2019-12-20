#include "yacx_Kernel.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "KernelUtils.h"
#include "../../include/yacx/Kernel.hpp"
#include "../../include/yacx/KernelArgs.hpp"
#include "../../include/yacx/Device.hpp"
#include "../../include/yacx/KernelTime.hpp"

#include <string>

using yacx::Kernel, yacx::KernelArg, yacx::KernelArgs, yacx::KernelTime, yacx::Device, jni::KernelArgJNI;

void Java_yacx_Kernel_configureInternal(JNIEnv *env, jobject obj, jint jgrid0, jint jgrid1, jint jgrid2, jint jblock0, jint jblock1, jint jblock2)
{
    BEGIN_TRY
        CHECK_BIGGER(jgrid0, 0, "illegal size for grid0", )
        CHECK_BIGGER(jgrid1, 0, "illegal size for grid1", )
        CHECK_BIGGER(jgrid2, 0, "illegal size for grid2", )
        CHECK_BIGGER(jblock0, 0, "illegal size for block0", )
        CHECK_BIGGER(jblock1, 0, "illegal size for block1", )
        CHECK_BIGGER(jblock2, 0, "illegal size for block2", )

        auto kernelPtr = getHandle<Kernel>(env, obj);
    	CHECK_NULL(kernelPtr, )

        dim3 grid{static_cast<unsigned int> (jgrid0), static_cast<unsigned int> (jgrid1), static_cast<unsigned int> (jgrid2)};
        dim3 block{static_cast<unsigned int> (jblock0), static_cast<unsigned int> (jblock1), static_cast<unsigned int> (jblock2)};

        kernelPtr->configure(grid, block);
    END_TRY("configuring Kernel")
}

jobject Java_yacx_Kernel_launchInternal___3Lyacx_KernelArg_2(JNIEnv *env, jobject obj, jobjectArray jArgs)
{
    BEGIN_TRY
        CHECK_NULL(jArgs, NULL);

        auto kernelPtr = getHandle<Kernel>(env, obj);
        CHECK_NULL(kernelPtr, NULL)

        auto args = getArguments(env, jArgs);
        if (args.empty()) return NULL;

        auto kernelTimePtr = kernelPtr->launch(KernelArgs{args});

        return createJavaKernelTime(env, &kernelTimePtr);
    END_TRY("launching Kernel")
}

jobject Java_yacx_Kernel_launchInternal__Lyacx_Device_2_3Lyacx_KernelArg_2(JNIEnv *env, jobject obj, jobject jdevice, jobjectArray jArgs)
{
    BEGIN_TRY
        CHECK_NULL(jArgs, NULL);
        CHECK_NULL(jdevice, NULL);

        auto kernelPtr = getHandle<Kernel>(env, obj);
        CHECK_NULL(kernelPtr, NULL);
        auto devicePtr = getHandle<Device>(env, jdevice);
        CHECK_NULL(devicePtr, NULL);

        auto args = getArguments(env, jArgs);
        if (args.empty()) return NULL;

        return launchInternal(env, kernelPtr, devicePtr, args);
    END_TRY("launching Kernel on specific device")
}

jobject Java_yacx_Kernel_launchInternal__Ljava_lang_String_2_3Lyacx_KernelArg_2(JNIEnv *env, jobject obj, jstring jdevicename, jobjectArray jArgs)
{
    BEGIN_TRY
        CHECK_NULL(jdevicename, NULL)
        CHECK_NULL(jArgs, NULL);

        auto kernelPtr = getHandle<Kernel>(env, obj);
        CHECK_NULL(kernelPtr, NULL);
        auto devicenamePtr = env->GetStringUTFChars(jdevicename, nullptr);
        std::string devicename{devicenamePtr};

        Device device{devicename};

        env->ReleaseStringUTFChars(jdevicename, devicenamePtr);

        auto args = getArguments(env, jArgs);
        if (args.empty()) return NULL;

        auto kernelTimePtr = kernelPtr->launch(KernelArgs{args}, device);

        return createJavaKernelTime(env, &kernelTimePtr);
    END_TRY("launching Kernel")
}
