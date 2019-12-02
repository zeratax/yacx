#include "Options.h"
#include "Handle.h"
#include "../../include/cudaexecutor/Options.hpp"
#include "../../include/cudaexecutor/Logger.hpp"

using cudaexecutor::Options;

jobject Java_Options_createOptions (JNIEnv* env, jclass cls){
    BEGIN_TRY
        auto optionPtr = new Options{};

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, optionPtr);

        return obj;
    END_TRY("creating Options")
}

void Java_Options_insertInternal__Ljava_lang_String_2 (JNIEnv* env, jobject obj, jstring joption){
    BEGIN_TRY
        auto optionPtr = env->GetStringUTFChars(joption, nullptr);

        auto optionsPtr = getHandle<Options>(env, obj);
        optionsPtr->insert(optionPtr);

        env->ReleaseStringUTFChars(joption, optionPtr);
    END_TRY("inserting Option")
}

void Java_Options_insertInternal__Ljava_lang_String_2Ljava_lang_String_2 (JNIEnv* env, jobject obj, jstring jname, jstring jvalue){
    BEGIN_TRY
        auto namePtr = env->GetStringUTFChars(jname, nullptr);
        auto valuePtr = env->GetStringUTFChars(jvalue, nullptr);

        auto optionsPtr = getHandle<Options>(env, obj);
        optionsPtr->insert(namePtr, valuePtr);

        env->ReleaseStringUTFChars(jname, namePtr);
        env->ReleaseStringUTFChars(jvalue, valuePtr);
    END_TRY("inserting Option")
}

jint Java_Options_getSize (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto optionPtr = getHandle<Options>(env, obj);
        auto size = optionPtr->numOptions();

        return size;
    END_TRY("getting size of Options")
}

jobjectArray Java_Options_options (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto optionPtr = getHandle<Options>(env, obj);
        auto size = optionPtr->numOptions();
        auto options = optionPtr->options();

        return createStringArray(env, options, size);
    END_TRY("getting Strings of Options")
}