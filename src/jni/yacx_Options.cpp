#include "yacx_Options.h"
#include "Handle.h"
#include "../../include/yacx/Options.hpp"
#include "../../include/yacx/Logger.hpp"

using yacx::Options;

jobject Java_yacx_Options_createOptions (JNIEnv* env, jclass cls){
    BEGIN_TRY
        auto optionPtr = new Options{};

        return createJNIObject(env, cls, optionPtr);
    END_TRY("creating Options")
}

void Java_yacx_Options_insertInternal__Ljava_lang_String_2 (JNIEnv* env, jobject obj, jstring joption){
    BEGIN_TRY
        CHECK_NULL(joption, )

        auto optionPtr = env->GetStringUTFChars(joption, nullptr);

        auto optionsPtr = getHandle<Options>(env, obj);
        CHECK_NULL(optionsPtr, );
        optionsPtr->insert(optionPtr);

        env->ReleaseStringUTFChars(joption, optionPtr);
    END_TRY("inserting Option")
}

void Java_yacx_Options_insertInternal__Ljava_lang_String_2Ljava_lang_String_2 (JNIEnv* env, jobject obj, jstring jname, jstring jvalue){
    BEGIN_TRY
        CHECK_NULL(jname, )
        CHECK_NULL(jvalue, )

        auto namePtr = env->GetStringUTFChars(jname, nullptr);
        auto valuePtr = env->GetStringUTFChars(jvalue, nullptr);

        auto optionsPtr = getHandle<Options>(env, obj);
        CHECK_NULL(optionsPtr, );
        optionsPtr->insert(namePtr, valuePtr);

        env->ReleaseStringUTFChars(jname, namePtr);
        env->ReleaseStringUTFChars(jvalue, valuePtr);
    END_TRY("inserting Option")
}

jint Java_yacx_Options_getSize (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto optionPtr = getHandle<Options>(env, obj);
    	CHECK_NULL(optionPtr, 0);
        auto size = optionPtr->numOptions();

        return size;
    END_TRY("getting size of Options")
}

jobjectArray Java_yacx_Options_getOptions (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto optionPtr = getHandle<Options>(env, obj);
    	CHECK_NULL(optionPtr, NULL);
        auto size = optionPtr->numOptions();
        auto options = optionPtr->content();

        return createStringArray(env, options, size);
    END_TRY("getting Strings of Options")
}
