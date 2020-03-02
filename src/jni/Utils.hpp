#pragma once

#include "../../include/yacx/Logger.hpp"
#include "Handle.h"
#include <vector>

jclass getClass(JNIEnv *env, const char *name);

jobjectArray createStringArray(JNIEnv *env, const char **stringArray, int size);

std::string getStaticJString(JNIEnv* env, jclass cls, const char* attributeName);

std::vector<std::string> jStringsToVector(JNIEnv* env, jobjectArray jstringArray);

#define CHECK_NULL(object, returnValue)                                        \
  if (object == NULL) {                                                        \
    logger(yacx::loglevel::ERROR)                                              \
        << "[JNI ERROR] NullPointerException in file:[" << __FILE__ << ":"     \
        << __LINE__ - 2 << "]";                                                \
                                                                               \
    jclass cls = getClass(env, "java/lang/NullPointerException");              \
                                                                               \
    if (cls)                                                                   \
      env->ThrowNew(cls, "");                                                  \
                                                                               \
    return returnValue;                                                        \
  }

#define CHECK_BIGGER(object, than, message, returnValue)                       \
  if (object <= than) {                                                        \
    logger(yacx::loglevel::ERROR)                                              \
        << "[JNI ERROR] IllegalArgumentException (" << message                 \
        << ") in file:[" << __FILE__ << ":" << __LINE__ - 2 << "]";            \
                                                                               \
    jclass cls = getClass(env, "java/lang/IllegalArgumentException");          \
                                                                               \
    if (cls)                                                                   \
      env->ThrowNew(cls, message);                                             \
                                                                               \
    return returnValue;                                                        \
  }

#define BEGIN_TRY try {

#define END_TRY(message)                                                       \
  }                                                                            \
  catch (const std::exception &err) {                                          \
    logger(yacx::loglevel::ERROR)                                              \
        << "Executor failure while " << message << ": " << err.what();         \
                                                                               \
    jclass cls = getClass(env, "yacx/ExecutorFailureException");               \
                                                                               \
    if (cls)                                                                   \
      env->ThrowNew(cls, (std::string("Executor failure while ") + message +   \
                          ": " + err.what())                                   \
                             .c_str());                                        \
  }                                                                            \
  catch (...) {                                                                \
    logger(yacx::loglevel::ERROR) << "Executor failure while " << message;     \
                                                                               \
    jclass cls = getClass(env, "yacx/ExecutorFailureException");               \
                                                                               \
    if (cls)                                                                   \
      env->ThrowNew(                                                           \
          cls, (std::string("Executor failure while ") + message).c_str());    \
  }

#define END_TRY_R(message, returnValue)                                        \
  }                                                                            \
  catch (const std::exception &err) {                                          \
    logger(yacx::loglevel::ERROR)                                              \
        << "Executor failure while " << message << ":" << err.what();          \
                                                                               \
    jclass cls = getClass(env, "yacx/ExecutorFailureException");               \
                                                                               \
    if (cls)                                                                   \
      env->ThrowNew(cls, (std::string("Executor failure while ") + message +   \
                          ": " + err.what())                                   \
                             .c_str());                                        \
    return returnValue;                                                        \
  }                                                                            \
  catch (...) {                                                                \
    logger(yacx::loglevel::ERROR) << "Executor failure while " << message;     \
                                                                               \
    jclass cls = getClass(env, "yacx/ExecutorFailureException");               \
                                                                               \
    if (cls)                                                                   \
      env->ThrowNew(                                                           \
          cls, (std::string("Executor failure while ") + message).c_str());    \
    return returnValue;                                                        \
  }

#define CTYPE getStaticJString(env, cls, "cType")