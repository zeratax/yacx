#include "DoubleArg.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/yacx/KernelArgs.hpp"

using yacx::KernelArg, jni::KernelArgJNI;

jobject JNICALL Java_DoubleArg_createValue(JNIEnv* env, jclass cls, jdouble jvalue){
	BEGIN_TRY
		cls = getClass(env, "KernelArg");
		if (cls == NULL) return NULL;

		KernelArgJNI* kernelArgPtr = new KernelArgJNI{&jvalue, sizeof(jdouble), false, false, false};

		return createJNIObject(env, cls, kernelArgPtr);
	END_TRY("creating DoubleValueArg")
}

jobject Java_DoubleArg_createInternal (JNIEnv* env, jclass cls, jdoubleArray jarray, jboolean jdownload){
    BEGIN_TRY
        CHECK_NULL(jarray, NULL)

        auto arrayPtr = env->GetDoubleArrayElements(jarray, NULL);
        auto arrayLength = env->GetArrayLength(jarray);

        CHECK_BIGGER(arrayLength, 0, "illegal array length", NULL)

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{arrayPtr, arrayLength * sizeof(jdouble), jdownload, true, true};

        env->ReleaseDoubleArrayElements(jarray, arrayPtr, JNI_ABORT);

        return createJNIObject(env, cls, kernelArgPtr);
    END_TRY("creating DoubleArg")
}

jobject Java_DoubleArg_createOutputInternal (JNIEnv* env, jclass cls, jint jarrayLength){
    BEGIN_TRY
        CHECK_BIGGER(jarrayLength, 0, "illegal array length", NULL)

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{NULL, static_cast<size_t> (jarrayLength) * sizeof(jdouble), true, false, true};

    	return createJNIObject(env, cls, kernelArgPtr);
    END_TRY("creating DoubleArg")
}

jdoubleArray Java_DoubleArg_asDoubleArray (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
    	CHECK_NULL(kernelArgJNIPtr, NULL)
        auto data = kernelArgJNIPtr->getHostData();
        auto dataSize = kernelArgJNIPtr->kernelArgPtr()->size();

        auto res = env->NewDoubleArray(dataSize / sizeof(jdouble));

        CHECK_NULL(res, NULL)

        env->SetDoubleArrayRegion(res, 0, dataSize / sizeof(jdouble),
                                   reinterpret_cast<const jdouble*>(data));
        return res;
    END_TRY("getting DoubleArg-content")
}

