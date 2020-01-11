#include "yacx_LongArg.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/yacx/KernelArgs.hpp"

using yacx::KernelArg, jni::KernelArgJNI;

jobject JNICALL Java_yacx_LongArg_createValue(JNIEnv* env, jclass cls, jlong jvalue){
	BEGIN_TRY
		cls = getClass(env, "yacx/KernelArg");
		if (cls == NULL) return NULL;

		KernelArgJNI* kernelArgPtr = new KernelArgJNI{&jvalue, sizeof(jlong), false, false, false};

		return createJNIObject(env, cls, kernelArgPtr);
	END_TRY("creating LongValueArg")
}

jobject Java_yacx_LongArg_createInternal (JNIEnv* env, jclass cls, jlongArray jarray, jboolean jdownload){
    BEGIN_TRY
        CHECK_NULL(jarray, NULL)

        auto arrayPtr = env->GetLongArrayElements(jarray, NULL);
        auto arrayLength = env->GetArrayLength(jarray);

        CHECK_BIGGER(arrayLength, 0, "illegal array length", NULL)

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{arrayPtr, arrayLength * sizeof(jlong), jdownload, true, true};

        env->ReleaseLongArrayElements(jarray, arrayPtr, JNI_ABORT);

        return createJNIObject(env, cls, kernelArgPtr);
    END_TRY("creating LongArg")
}

jlongArray Java_yacx_LongArg_asLongArray (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
    	CHECK_NULL(kernelArgJNIPtr, NULL)
        auto data = kernelArgJNIPtr->getHostData();
        auto dataSize = kernelArgJNIPtr->kernelArgPtr()->size();

        auto res = env->NewLongArray(dataSize / sizeof(jlong));

        CHECK_NULL(res, NULL)

        env->SetLongArrayRegion(res, 0, dataSize / sizeof(jlong),
                                   reinterpret_cast<const jlong*>(data));
        return res;
    END_TRY("getting LongArg-content")
}
