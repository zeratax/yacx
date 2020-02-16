#pragma once

#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/yacx/Kernel.hpp"
#include "../../include/yacx/KernelArgs.hpp"
#include "../../include/yacx/Devices.hpp"
#include "../../include/yacx/KernelTime.hpp"

using yacx::Kernel, yacx::KernelArg, yacx::KernelArgs, yacx::KernelTime, yacx::Device, jni::KernelArgJNI;

std::vector<KernelArg> getArguments(JNIEnv* env, jobjectArray jArgs);

jobject createJavaKernelTime(JNIEnv* env, KernelTime* kernelTimePtr);

jobject launchInternal(JNIEnv *env, Kernel* kernelptr, Device* devicePtr, std::vector<KernelArg> args);
