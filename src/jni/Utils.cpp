#include "Utils.hpp"
#include <stdlib.h>

jclass getClass(JNIEnv* env, const char* name) {
    jclass cls = env->FindClass(name);

    if (!cls) {
        logger(yacx::loglevel::ERROR) << "[JNI ERROR] Cannot find the " << name << " class";

        cls = env->FindClass("java/lang/ClassNotFoundException");

        if (!cls) {
            logger(yacx::loglevel::ERROR) << "[JNI ERROR] Cannot find java.lang.ClassNotFoundException";
            exit(1);
        }

        env->ThrowNew(cls, name);
        exit(1);
    }

    return cls;
}

jobjectArray createStringArray(JNIEnv* env, const char** stringArray, int size) {
    jclass stringCls = getClass(env, "java/lang/String");

    auto res = (jobjectArray) env->NewObjectArray(size, stringCls, env->NewStringUTF(""));
    CHECK_NULL(res)

    for(int i = 0; i < size; i++){
        auto jstring = env->NewStringUTF(stringArray[i]);
        env->SetObjectArrayElement(res, i, jstring);
    }

    return res;
}