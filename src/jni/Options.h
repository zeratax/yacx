/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class Options */

#ifndef _Included_Options
#define _Included_Options
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     Options
 * Method:    createOptions
 * Signature: ()LOptions;
 */
JNIEXPORT jobject JNICALL Java_Options_createOptions
  (JNIEnv *, jclass);

/*
 * Class:     Options
 * Method:    insertInternal
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_Options_insertInternal__Ljava_lang_String_2
  (JNIEnv *, jobject, jstring);

/*
 * Class:     Options
 * Method:    insertInternal
 * Signature: (Ljava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_Options_insertInternal__Ljava_lang_String_2Ljava_lang_String_2
  (JNIEnv *, jobject, jstring, jstring);

/*
 * Class:     Options
 * Method:    getSize
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_Options_getSize
  (JNIEnv *, jobject);

/*
 * Class:     Options
 * Method:    options
 * Signature: ()[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_Options_options
  (JNIEnv *, jobject);

#ifdef __cplusplus
}
#endif
#endif