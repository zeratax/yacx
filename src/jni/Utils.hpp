#pragma once

#include "../../include/cudaexecutor/Logger.hpp"
#include <jni.h>

#define CHECK_NULL(object)                                                                                       \
    if (object == NULL) {                                                                                        \
        logger(cudaexecutor::loglevel::ERROR) << "[JNI ERROR] NullPointerException in file:[" << __FILE__        \
            << ":" << __LINE__-2 << "]";                                                                         \
                                                                                                                 \
        jclass cls = getClass(env, "ExecutorFailureException");                                                  \
                                                                                                                 \
        env->ThrowNew(cls, "");                                                                                  \
    }

#define CHECK_BIGGER(object, than, message)                                                                      \
    if (object <= than) {                                                                                        \
        logger(cudaexecutor::loglevel::ERROR) << "[JNI ERROR] IllegalArgumentException (" << message             \
            << ") in file:[" << __FILE__  << ":" << __LINE__-2 << "]";                                           \
                                                                                                                 \
        jclass cls = getClass(env, "java/lang/IllegalArgumentException");                                        \
                                                                                                                 \
        env->ThrowNew(cls, message);                                                                             \
    }

#define BEGIN_TRY try {

#define END_TRY(message)                                                                                         \
     } catch (const std::exception &err) {                                                                       \
        logger(cudaexecutor::loglevel::ERROR) << "Executor failure while " << message <<  ":"  << err.what();    \
                                                                                                                 \
        jclass cls = getClass(env, "ExecutorFailureException");                                                  \
                                                                                                                 \
        env->ThrowNew(cls, (std::string("Executor failure while ") + message + ": " + err.what()).c_str());      \
    } catch (...) {                                                                                              \
        logger(cudaexecutor::loglevel::ERROR) << "Executor failure while " << message;                           \
                                                                                                                 \
        jclass cls = getClass(env, "ExecutorFailureException");                                                  \
                                                                                                                 \
        env->ThrowNew(cls, (std::string("Executor failure while ") + message).c_str());                          \
    }

jclass getClass(JNIEnv* env, const char* name);

jobjectArray createStringArray(JNIEnv* env, const char** stringArray, int size);
