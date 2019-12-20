#include "yacx_IntArg.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/yacx/KernelArgs.hpp"

using yacx::KernelArg, jni::KernelArgJNI;

jobject JNICALL Java_yacx_IntArg_createValue(JNIEnv* env, jclass cls, jint jvalue){
	BEGIN_TRY
		cls = getClass(env, "yacx/KernelArg");
		if (cls == NULL) return NULL;

		KernelArgJNI* kernelArgPtr = new KernelArgJNI{&jvalue, sizeof(jint), false, false, false};

		return createJNIObject(env, cls, kernelArgPtr);
	END_TRY("creating IntValueArg")
}


JNIEXPORT jobject JNICALL Java_yacx_IntArg_create (JNIEnv * env, jclass cls, jobject obj, jboolean jdownload) {
	BEGIN_TRY

		auto jarray = Java_yacx_IntArg_asIntArray(env, obj);
		auto arrayPtr = env->GetIntArrayElements(jarray, NULL);
		auto arrayLength = env->GetArrayLength(jarray);

		CHECK_BIGGER(arrayLength, 0, "illegal array length", NULL);

		KernelArgJNI* kernelArgPtr = new KernelArgJNI{arrayPtr, arrayLength * sizeof(jint), jdownload, true, true};

    env->ReleaseIntArrayElements(jarray, arrayPtr, JNI_ABORT);

    return createJNIObject(env, cls, kernelArgPtr);
	END_TRY("creating IntArg")
}

jobject Java_yacx_IntArg_createInternal (JNIEnv* env, jclass cls, jintArray jarray, jboolean jdownload){
    BEGIN_TRY
        CHECK_NULL(jarray, NULL)

        auto arrayPtr = env->GetIntArrayElements(jarray, NULL);
        auto arrayLength = env->GetArrayLength(jarray);

        CHECK_BIGGER(arrayLength, 0, "illegal array length", NULL)

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{arrayPtr, arrayLength * sizeof(jint), jdownload, true, true};

        env->ReleaseIntArrayElements(jarray, arrayPtr, JNI_ABORT);

        return createJNIObject(env, cls, kernelArgPtr);
    END_TRY("creating IntArg")
}

jobject Java_yacx_IntArg_createOutputInternal (JNIEnv* env, jclass cls, jint jarrayLength){
    BEGIN_TRY
        CHECK_BIGGER(jarrayLength, 0, "illegal array length", NULL)

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{NULL, static_cast<size_t> (jarrayLength) * sizeof(jint), true, false, true};

    	return createJNIObject(env, cls, kernelArgPtr);
    END_TRY("creating IntArg")
}

jintArray Java_yacx_IntArg_asIntArray (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
    	CHECK_NULL(kernelArgJNIPtr, NULL)
        auto data = kernelArgJNIPtr->getHostData();
        auto dataSize = kernelArgJNIPtr->kernelArgPtr()->size();

        auto res = env->NewIntArray(dataSize / sizeof(jint));

        CHECK_NULL(res, NULL)

        env->SetIntArrayRegion(res, 0, dataSize / sizeof(jint),
                                   reinterpret_cast<const jint*>(data));
        return res;
    END_TRY("getting IntArg-content")
}
