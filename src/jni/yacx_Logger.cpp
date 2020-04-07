#include "yacx_Logger.h"
#include "Handle.h"
#include "Utils.hpp"
#include "../../include/yacx/Logger.hpp"

using yacx::Logger;

void Java_yacx_Logger_setLogLevel (JNIEnv* env, jclass, jobject jloglevel){
  BEGIN_TRY
      jclass clsloglevel = getClass(env, "yacx/Logger$LogLevel");
      CHECK_NULL(clsloglevel, )

      jmethodID ordinal = env->GetMethodID(clsloglevel, "ordinal", "()I");
      CHECK_NULL(ordinal, )

      jint loglevel = env->CallIntMethod(jloglevel, ordinal);
      CHECK_BIGGER(loglevel, -1, "unknown loglevel", )

      Logger::getInstance().set_loglimit(static_cast<yacx::loglevel> (loglevel));
  END_TRY("setting loglevel")
}

void Java_yacx_Logger_setCout (JNIEnv* env, jclass, jboolean jcout){
  BEGIN_TRY
      Logger::getInstance().set_cout(jcout);
  END_TRY("setting/unsetting cout as ouput for logger")
}

void Java_yacx_Logger_setCerr (JNIEnv* env, jclass, jboolean jcerr){
  BEGIN_TRY
    Logger::getInstance().set_cerr(jcerr);
  END_TRY("setting/unsetting cerr as ouput for logger")
}

void Java_yacx_Logger_setLogfile (JNIEnv* env, jclass, jstring jfilename){
  BEGIN_TRY
    CHECK_NULL(jfilename, )

    auto filename = env->GetStringUTFChars(jfilename, NULL);
    CHECK_NULL(filename, )

    Logger::getInstance().set_logfile(filename);

    env->ReleaseStringUTFChars(jfilename, filename);
  END_TRY("setting logfile")
}