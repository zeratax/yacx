#include "yacx_BooleanArg.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/yacx/KernelArgs.hpp"

using yacx::KernelArg, jni::KernelArgJNI;

jobject JNICALL Java_yacx_BooleanArg_createValue(JNIEnv* env, jclass cls, jboolean jvalue){
	BEGIN_TRY
		jclass clsKernelArg = getClass(env, "yacx/KernelArg");
		if (clsKernelArg == NULL) return NULL;

		KernelArgJNI* kernelArgPtr = new KernelArgJNI{&jvalue, sizeof(jboolean), false, false, false, CTYPE};

		return createJNIObject(env, clsKernelArg, kernelArgPtr);
	END_TRY_R("creating BooleanValueArg", NULL)
}

jobject Java_yacx_BooleanArg_createInternal (JNIEnv* env, jclass cls, jbooleanArray jarray, jboolean jdownload){
    BEGIN_TRY
        CHECK_NULL(jarray, NULL)

        auto arrayPtr = env->GetBooleanArrayElements(jarray, NULL);
        auto arrayLength = env->GetArrayLength(jarray);

        CHECK_BIGGER(arrayLength, 0, "illegal array length", NULL)

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{arrayPtr, arrayLength * sizeof(jboolean), jdownload, true, true, CTYPE + "*"};

        env->ReleaseBooleanArrayElements(jarray, arrayPtr, JNI_ABORT);

        return createJNIObject(env, cls, kernelArgPtr);
    END_TRY_R("creating BooleanArg", NULL)
}

jbooleanArray Java_yacx_BooleanArg_asBooleanArray (JNIEnv* env, jobject obj){
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
    END_TRY_R("getting BooleanArg-content", NULL)
}
