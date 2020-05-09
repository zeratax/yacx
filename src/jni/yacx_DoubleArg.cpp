#include "yacx_DoubleArg.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/yacx/KernelArgs.hpp"

using yacx::KernelArg, jni::KernelArgJNI;

jobject JNICALL Java_yacx_DoubleArg_createValue(JNIEnv* env, jclass cls, jdouble jvalue){
	BEGIN_TRY
		jclass clsKernelArg = getClass(env, "yacx/KernelArg");
		if (clsKernelArg == NULL) return NULL;

		KernelArgJNI* kernelArgPtr = new KernelArgJNI{sizeof(jdouble), false, false, false, CTYPE};
		*(static_cast<jdouble*> (kernelArgPtr->getHostData())) = jvalue;

		return createJNIObject(env, clsKernelArg, kernelArgPtr);
	END_TRY_R("creating DoubleValueArg", NULL)
}

jobject Java_yacx_DoubleArg_createInternal (JNIEnv* env, jclass cls, jdoubleArray jarray, jboolean jdownload){
    BEGIN_TRY
        CHECK_NULL(jarray, NULL)

        auto arrayLength = env->GetArrayLength(jarray);
        CHECK_BIGGER(arrayLength, 0, "illegal array length", NULL)

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{arrayLength * sizeof(jdouble), static_cast<bool>(jdownload), true, true, CTYPE + "*"};
        env->GetDoubleArrayRegion(jarray, 0, arrayLength, static_cast<jdouble*> (kernelArgPtr->getHostData()));

        return createJNIObject(env, cls, kernelArgPtr);
    END_TRY_R("creating DoubleArg", NULL)
}

jdoubleArray Java_yacx_DoubleArg_asDoubleArray (JNIEnv* env, jobject obj){
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
    END_TRY_R("getting DoubleArg-content", NULL)
}
