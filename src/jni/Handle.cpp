#include <jni.h>

#include "Handle.h"

jfieldID getHandleField(JNIEnv* env, jobject obj)
{
    auto c = env->GetObjectClass(obj);
    return env->GetFieldID(c, "nativeHandle", "J");
}

void clearHandle(JNIEnv* env, jobject obj)
{
    env->SetLongField(obj, getHandleField(env, obj), 0);
}

jobjectArray createStringArray(JNIEnv* env, const char** stringArray, int size){
    auto stringCls = env->FindClass("java/lang/String");

    auto res = (jobjectArray) env->NewObjectArray(size, stringCls, env->NewStringUTF(""));
    if (res == nullptr) return nullptr;

    for(int i = 0; i < size; i++){
        auto jstring = env->NewStringUTF(stringArray[i]);
        env->SetObjectArrayElement(res, i, jstring);
    }

    return res;
}