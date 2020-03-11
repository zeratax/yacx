/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class yacx_CProgram */

#ifndef _Included_yacx_CProgram
#define _Included_yacx_CProgram
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     yacx_CProgram
 * Method:    getTypes
 * Signature: ([Lyacx/KernelArg;)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_yacx_CProgram_getTypes
  (JNIEnv *, jclass, jobjectArray);

/*
 * Class:     yacx_CProgram
 * Method:    createInternal
 * Signature: (Ljava/lang/String;Ljava/lang/String;[Ljava/lang/String;Ljava/lang/String;Lyacx/Options;)Lyacx/CProgram;
 */
JNIEXPORT jobject JNICALL Java_yacx_CProgram_createInternal
  (JNIEnv *, jclass, jstring, jstring, jobjectArray, jstring, jobject);

/*
 * Class:     yacx_CProgram
 * Method:    execute
 * Signature: ([Lyacx/KernelArg;)V
 */
JNIEXPORT void JNICALL Java_yacx_CProgram_execute
  (JNIEnv *, jobject, jobjectArray);

#ifdef __cplusplus
}
#endif
#endif
