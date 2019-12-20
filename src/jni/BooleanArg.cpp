#include "BooleanArg.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/yacx/KernelArgs.hpp"

using yacx::KernelArg, jni::KernelArgJNI;

jobject JNICALL Java_BooleanArg_createValue(JNIEnv* env, jclass cls, jboolean jvalue){
	BEGIN_TRY
		cls = getClass(env, "KernelArg");
		if (cls == NULL) return NULL;

		KernelArgJNI* kernelArgPtr = new KernelArgJNI{&jvalue, sizeof(jboolean), false, false, false};

		return createJNIObject(env, cls, kernelArgPtr);
	END_TRY("creating BooleanValueArg")
}


jobject Java_BooleanArg_create(JNIEnv *env, jclass cls, jobject obj, jboolean jdownload) {
	BEGIN_TRY

		auto jarray = Java_BooleanArg_asBooleanArray(env, obj);
		auto arrayPtr = env->GetBooleanArrayElements(jarray, NULL);
		auto arrayLength = env->GetArrayLength(jarray);

	CHECK_BIGGER(arrayLength, 0, "illegal array length", NULL);

	KernelArgJNI* kernelArgPtr = new KernelArgJNI{arrayPtr, arrayLength * sizeof(jboolean), true, true};

	env->ReleaseBooleanArrayElements(jarray, arrayPtr, JNI_ABORT);

	return createJNIObject(env, cls, kernelArgPtr);
	END_TRY("creating BooleanArg")
}

jobject Java_BooleanArg_createInternal (JNIEnv* env, jclass cls, jbooleanArray jarray, jboolean jdownload){
    BEGIN_TRY
        CHECK_NULL(jarray, NULL)

        auto arrayPtr = env->GetBooleanArrayElements(jarray, NULL);
        auto arrayLength = env->GetArrayLength(jarray);

        CHECK_BIGGER(arrayLength, 0, "illegal array length", NULL)

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{arrayPtr, arrayLength * sizeof(jboolean), jdownload, true, true};

        env->ReleaseBooleanArrayElements(jarray, arrayPtr, JNI_ABORT);

        return createJNIObject(env, cls, kernelArgPtr);
    END_TRY("creating BooleanArg")
}

jobject Java_BooleanArg_createOutputInternal (JNIEnv* env, jclass cls, jint jarrayLength){
    BEGIN_TRY
        CHECK_BIGGER(jarrayLength, 0, "illegal array length", NULL)

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{NULL, static_cast<size_t> (jarrayLength) * sizeof(jboolean), true, false, true};

    	return createJNIObject(env, cls, kernelArgPtr);
    END_TRY("creating BooleanArg")
}

jbooleanArray Java_BooleanArg_asBooleanArray (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
    	CHECK_NULL(kernelArgJNIPtr, NULL)
        auto data = kernelArgJNIPtr->getHostData();
        auto dataSize = kernelArgJNIPtr->kernelArgPtr()->size();

        auto res = env->NewBooleanArray(dataSize / sizeof(jboolean));

        CHECK_NULL(res, NULL)

        env->SetBooleanArrayRegion(res, 0, dataSize / sizeof(jboolean),
                                   reinterpret_cast<const jboolean*>(data));
        return res;
    END_TRY("getting BooleanArg-content")
}
