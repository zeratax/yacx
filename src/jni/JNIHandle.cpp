#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#include <jni.h>
#pragma GCC diagnostic pop
#include "JNIHandle.h"
#include "Handle.h"

void JNICALL Java_JNIHandle_dispose(JNIEnv *env, jobject obj)
{
    auto ptr = getHandle(env, obj);
    clearHandle(env, obj);
    delete ptr;
}