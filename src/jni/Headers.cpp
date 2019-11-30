#include "Headers.h"
#include "Handle.h"
#include "../../include/cudaexecutor/Logger.hpp"
#include "../../include/cudaexecutor/Headers.hpp"

using cudaexecutor::loglevel, cudaexecutor::Headers;

jobject Java_Headers_createHeaders (JNIEnv* env, jclass cls){
    BEGIN_TRY
        auto headersPtr = new Headers{};

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, headersPtr);

        return obj;
    END_TRY("creating headers")
}

void Java_Headers_insertInternal__Ljava_lang_String_2 (JNIEnv* env, jobject obj, jstring jheaderPath){
    BEGIN_TRY
        auto headersPtr = getHandle<Headers>(env, obj);

        auto headerPathPtr = env->GetStringUTFChars(jheaderPath, nullptr);

        headersPtr->insert(headerPathPtr);

        env->ReleaseStringUTFChars(jheaderPath, headerPathPtr);
    END_TRY("inserting header")
}

void Java_Headers_insertInternal___3Ljava_lang_String_2 (JNIEnv* env, jobject obj, jobjectArray jheaderPathArray){
    BEGIN_TRY
        auto headersPtr = getHandle<Headers>(env, obj);

        int length = env->GetArrayLength(jheaderPathArray);

        for (int i = 0; i < length; i++) {
            auto jheaderPath = static_cast<jstring> (env->GetObjectArrayElement(jheaderPathArray, i));

            auto headerPathPtr = env->GetStringUTFChars(jheaderPath, nullptr);

            headersPtr->insert(headerPathPtr);

            env->ReleaseStringUTFChars(jheaderPath, headerPathPtr);
        }
    END_TRY("inserting headers")
}

jint Java_Headers_getSize (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto headersPtr = getHandle<Headers>(env, obj);
        auto size = headersPtr->size();

        return size;
    END_TRY("getting size of headers")
}

jobjectArray Java_Headers_names (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto headersPtr = getHandle<Headers>(env, obj);
        auto size = headersPtr->size();
        auto names = headersPtr->names();

        return createStringArray(env, names, size);
    END_TRY("getting names of headers")
}

jobjectArray Java_Headers_content (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto headersPtr = getHandle<Headers>(env, obj);
        auto size = headersPtr->size();
        auto content = headersPtr->content();

        return createStringArray(env, content, size);
    END_TRY("getting content of headers")
}