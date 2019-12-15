#include <jni.h>

#include "Handle.h"

jfieldID getHandleField(JNIEnv* env, jobject obj)
{
    auto c = env->GetObjectClass(obj);
    return env->GetFieldID(c, "nativeHandle", "J");
}

void clearHandle(JNIEnv* env, jobject obj)
{
    env->SetLongField(obj, getHandleField(env, obj), 0);
}

jobject createJNIObject(JNIEnv* env, jclass cls, void* objectPtr){
	auto methodID = env->GetMethodID(cls, "<init>", "(J)V");
	return env->NewObject(cls, methodID, objectPtr);
}
