#include "KernelUtils.h"

std::vector<KernelArg> getArguments(JNIEnv* env, jobjectArray jArgs)
{
    CHECK_NULL(jArgs, {})

    auto argumentsLength = env->GetArrayLength(jArgs);

    CHECK_BIGGER(argumentsLength, 0, "illegal array length", {})

    std::vector<KernelArg> args;
    args.reserve(argumentsLength);

    for(int i = 0; i < argumentsLength; i++){
        auto jkernelArg = env->GetObjectArrayElement(jArgs, i);

        CHECK_NULL(jkernelArg, {})

        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, jkernelArg);
        CHECK_NULL(kernelArgJNIPtr, {})
        args.push_back(*kernelArgJNIPtr->kernelArgPtr());
    }

    return args;
}

jobject createJavaKernelTime(JNIEnv* env, KernelTime* kernelTimePtr){
    jclass cls = getClass(env, "yacx/KernelTime");
    if (cls == NULL) return NULL;

    auto methodID = env->GetMethodID(cls, "<init>", "(FFFF)V");
    auto obj = env->NewObject(cls, methodID, kernelTimePtr->upload, kernelTimePtr->download,
            kernelTimePtr->launch, kernelTimePtr->total);

    return obj;
}

jobject launchInternal(JNIEnv *env, Kernel* kernelPtr, Device& device, std::vector<KernelArg> args)
{
    auto kernelTimePtr = kernelPtr->launch(KernelArgs{args}, device);

    return createJavaKernelTime(env, &kernelTimePtr);
}
