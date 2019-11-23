#include "JNIHandle.h"
#include "Handle.h"
#include "../../include/cudaexecutor/JNIHandle.hpp"

using cudaexecutor::JNIHandle;

void JNICALL Java_JNIHandle_dispose(JNIEnv *env, jobject obj)
{
    auto ptr = getHandle<JNIHandle>(env, obj);
    clearHandle(env, obj);
    delete ptr;
}