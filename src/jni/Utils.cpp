#include "Utils.hpp"
#include <stdlib.h>

jclass getClass(JNIEnv* env, const char* name) {
    jclass cls = env->FindClass(name);

    if (!cls) {
        logger(yacx::loglevel::ERROR) << "[JNI ERROR] Cannot find the " << name << " class";

        cls = env->FindClass("java/lang/ClassNotFoundException");

        if (!cls) {
            logger(yacx::loglevel::ERROR) << "[JNI ERROR] Cannot find java.lang.ClassNotFoundException";
			return NULL;
        }

        env->ThrowNew(cls, name);

        return NULL;
    }

    return cls;
}

jobjectArray createStringArray(JNIEnv* env, const char** stringArray, int size) {
    jclass stringCls = getClass(env, "java/lang/String");

    if (!stringCls) return NULL;

    auto res = (jobjectArray) env->NewObjectArray(size, stringCls, env->NewStringUTF(""));
    CHECK_NULL(res, NULL)

    for(int i = 0; i < size; i++){
        auto jstring = env->NewStringUTF(stringArray[i]);
        env->SetObjectArrayElement(res, i, jstring);
    }

    return res;
}

std::vector<std::string> jStringsToVector(JNIEnv* env, jobjectArray jstringArray) {
    CHECK_NULL(jstringArray, {})

    auto argumentsLength = env->GetArrayLength(jstringArray);

    CHECK_BIGGER(argumentsLength, 0, "illegal array length", {})

    std::vector<std::string> args;
    args.reserve(argumentsLength);

    for(int i = 0; i < argumentsLength; i++){
        auto jString = static_cast<jstring> (env->GetObjectArrayElement(jstringArray, i));
        CHECK_NULL(jString, {})

        auto stringPtr = env->GetStringUTFChars(jString, nullptr);
        args.push_back(std::string(stringPtr));
        env->ReleaseStringUTFChars(jString, stringPtr);
    }

    return args;
}

std::string getStaticJString(JNIEnv* env, jclass cls, const char* attributeName) {
    jfieldID jfid = env->GetStaticFieldID(cls, attributeName, "Ljava/lang/String;");
    if (jfid == NULL){
        logger(yacx::loglevel::ERROR) << "[JNI ERROR] Cannot find attribute " << attributeName
        << " in class " << cls;
        throw std::runtime_error(std::string("Cannot find attribute ") + attributeName);
        return NULL;
    }
    
    jstring jString = static_cast<jstring> (env->GetStaticObjectField(cls, jfid));

    auto stringPtr = env->GetStringUTFChars(jString, nullptr);
    auto string = std::string(stringPtr);
    env->ReleaseStringUTFChars(jString, stringPtr);

    return string;
}