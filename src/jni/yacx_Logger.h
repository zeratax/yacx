/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class yacx_Logger */

#ifndef _Included_yacx_Logger
#define _Included_yacx_Logger
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     yacx_Logger
 * Method:    setLogLevel
 * Signature: (Lyacx/Logger/LogLevel;)V
 */
JNIEXPORT void JNICALL Java_yacx_Logger_setLogLevel
  (JNIEnv *, jclass, jobject);

/*
 * Class:     yacx_Logger
 * Method:    setCout
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_yacx_Logger_setCout
  (JNIEnv *, jclass, jboolean);

/*
 * Class:     yacx_Logger
 * Method:    setCerr
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL Java_yacx_Logger_setCerr
  (JNIEnv *, jclass, jboolean);

/*
 * Class:     yacx_Logger
 * Method:    setLogfile
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_yacx_Logger_setLogfile
  (JNIEnv *, jclass, jstring);

#ifdef __cplusplus
}
#endif
#endif
