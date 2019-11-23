#include "Options.h"
#include "Handle.h"
#include "../../include/cudaexecutor/Options.hpp"
#include "../../include/cudaexecutor/Logger.hpp"

using cudaexecutor::loglevel, cudaexecutor::Options;

void Java_Options_insert__Ljava_lang_String_2 (JNIEnv* env, jobject obj, jstring joption){
    try {
        auto optionPtr = env->GetStringUTFChars(joption, nullptr);

        auto optionsPtr = getHandle<Options>(env, obj);
        optionsPtr->insert(optionPtr);

        env->ReleaseStringUTFChars(joption, optionPtr);
    }catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while inserting Option");
    }
}

void Java_Options_insert__Ljava_lang_String_2Ljava_lang_String_2 (JNIEnv* env, jobject obj, jstring jname, jstring jvalue){
    try {
        auto namePtr = env->GetStringUTFChars(jname, nullptr);
        auto valuePtr = env->GetStringUTFChars(jvalue, nullptr);

        auto optionsPtr = getHandle<Options>(env, obj);
        optionsPtr->insert(namePtr, valuePtr);

        env->ReleaseStringUTFChars(jname, namePtr);
        env->ReleaseStringUTFChars(jvalue, valuePtr);
    }catch (...){
        jclass jClass = env->FindClass("ExecutorFailureException");

        if(!jClass)
            logger(loglevel::ERROR) << "[JNI ERROR] Cannot find the exception class";

        env->ThrowNew(jClass, "Executor failure while inserting Option");
    }
}
