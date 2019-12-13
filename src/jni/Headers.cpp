#include "Headers.h"
#include "Handle.h"
#include "../../include/yacx/Logger.hpp"
#include "../../include/yacx/Headers.hpp"

using yacx::Headers;

jobject Java_Headers_createHeaders (JNIEnv* env, jclass cls){
    BEGIN_TRY
        auto headersPtr = new Headers{};

        auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
        auto obj = env->NewObject(cls, methodID, headersPtr);

        return obj;
    END_TRY("creating headers")
}

void Java_Headers_insertInternal (JNIEnv* env, jobject obj, jobjectArray jheaderPathArray){
    BEGIN_TRY
        CHECK_NULL(jheaderPathArray);

        auto headersPtr = getHandle<Headers>(env, obj);

        int length = env->GetArrayLength(jheaderPathArray);

        CHECK_BIGGER(length, 0, "illegal array length")

        for (int i = 0; i < length; i++) {
            auto jheaderPath = static_cast<jstring> (env->GetObjectArrayElement(jheaderPathArray, i));

            CHECK_NULL(jheaderPath);

            auto headerPathPtr = env->GetStringUTFChars(jheaderPath, nullptr);

            headersPtr->insert(headerPathPtr);

            env->ReleaseStringUTFChars(jheaderPath, headerPathPtr);
        }
    END_TRY("inserting headers")
}

jint Java_Headers_getSize (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto headersPtr = getHandle<Headers>(env, obj);
        auto size = headersPtr->numHeaders();

        return size;
    END_TRY("getting size of headers")
}

jobjectArray Java_Headers_names (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto headersPtr = getHandle<Headers>(env, obj);
        auto size = headersPtr->numHeaders();
        auto names = headersPtr->names();

        return createStringArray(env, names, size);
    END_TRY("getting names of headers")
}

jobjectArray Java_Headers_content (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto headersPtr = getHandle<Headers>(env, obj);
        auto size = headersPtr->numHeaders();
        auto content = headersPtr->content();

        return createStringArray(env, content, size);
    END_TRY("getting content of headers")
}