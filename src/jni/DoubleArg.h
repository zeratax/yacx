/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class DoubleArg */

#ifndef _Included_DoubleArg
#define _Included_DoubleArg
#ifdef __cplusplus
extern "C" {
#endif
#undef DoubleArg_SIZE_BYTES
#define DoubleArg_SIZE_BYTES 8L
/*
 * Class:     DoubleArg
 * Method:    createValue
 * Signature: (D)LKernelArg;
 */
JNIEXPORT jobject JNICALL Java_DoubleArg_createValue
  (JNIEnv *, jclass, jdouble);

/*
 * Class:     DoubleArg
 * Method:    createInternal
 * Signature: ([DZ)LDoubleArg;
 */
JNIEXPORT jobject JNICALL Java_DoubleArg_createInternal
  (JNIEnv *, jclass, jdoubleArray, jboolean);

/*
 * Class:     DoubleArg
 * Method:    createOutputInternal
 * Signature: (I)LDoubleArg;
 */
JNIEXPORT jobject JNICALL Java_DoubleArg_createOutputInternal
  (JNIEnv *, jclass, jint);

/*
 * Class:     DoubleArg
 * Method:    asDoubleArray
 * Signature: ()[D
 */
JNIEXPORT jdoubleArray JNICALL Java_DoubleArg_asDoubleArray
  (JNIEnv *, jobject);

#ifdef __cplusplus
}
#endif
#endif
