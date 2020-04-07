#include "yacx_ShortArg.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "../../include/yacx/KernelArgs.hpp"

using yacx::KernelArg, jni::KernelArgJNI;

jobject JNICALL Java_yacx_ShortArg_createValue(JNIEnv* env, jclass cls, jshort jvalue){
	BEGIN_TRY
		jclass clsKernelArg = getClass(env, "yacx/KernelArg");
		if (clsKernelArg == NULL) return NULL;

		KernelArgJNI* kernelArgPtr = new KernelArgJNI{sizeof(jshort), false, false, false, CTYPE};
		*(static_cast<jshort*> (kernelArgPtr->getHostData())) = jvalue;

		return createJNIObject(env, clsKernelArg, kernelArgPtr);
	END_TRY_R("creating ShortValueArg", NULL)
}

jobject Java_yacx_ShortArg_createInternal (JNIEnv* env, jclass cls, jshortArray jarray, jboolean jdownload){
    BEGIN_TRY
        CHECK_NULL(jarray, NULL)

        auto arrayLength = env->GetArrayLength(jarray);
        CHECK_BIGGER(arrayLength, 0, "illegal array length", NULL)

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{arrayLength * sizeof(jshort), jdownload, true, true, CTYPE + "*"};
        env->GetShortArrayRegion(jarray, 0, arrayLength, static_cast<jshort*> (kernelArgPtr->getHostData()));

        return createJNIObject(env, cls, kernelArgPtr);
    END_TRY_R("creating ShortArg", NULL)
}

jshortArray Java_yacx_ShortArg_asShortArray (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
    	CHECK_NULL(kernelArgJNIPtr, NULL)
        auto data = kernelArgJNIPtr->getHostData();
        auto dataSize = kernelArgJNIPtr->kernelArgPtr()->size();

        auto res = env->NewShortArray(dataSize / sizeof(jshort));

        CHECK_NULL(res, NULL)

        env->SetShortArrayRegion(res, 0, dataSize / sizeof(jshort),
                                   reinterpret_cast<const jshort*>(data));
        return res;
    END_TRY_R("getting ShortArg-content", NULL)
}
