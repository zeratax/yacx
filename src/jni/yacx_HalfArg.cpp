#include "yacx_HalfArg.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "Half.hpp"
#include "../../include/yacx/KernelArgs.hpp"

using yacx::KernelArg, jni::KernelArgJNI;

jobject JNICALL Java_yacx_HalfArg_createValue(JNIEnv* env, jclass cls, jfloat jvalue){
	BEGIN_TRY
		cls = getClass(env, "yacx/KernelArg");
		if (cls == NULL) return NULL;

		KernelArgJNI* kernelArgPtr = new KernelArgJNI{&jvalue, sizeof(jfloat), false, false, false};

		return createJNIObject(env, cls, kernelArgPtr);
	END_TRY_R("creating HalfValueArg", NULL)
}

jobject Java_yacx_HalfArg_createInternal (JNIEnv* env, jclass cls, jfloatArray jarray, jboolean jdownload){
    BEGIN_TRY
        CHECK_NULL(jarray, NULL)

        auto arrayPtr = env->GetFloatArrayElements(jarray, NULL);
        auto arrayLength = env->GetArrayLength(jarray);

        CHECK_BIGGER(arrayLength, 0, "illegal array length", NULL)

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{arrayPtr, arrayLength * sizeof(jfloat), jdownload, true, true};

        env->ReleaseFloatArrayElements(jarray, arrayPtr, JNI_ABORT);

        return createJNIObject(env, cls, kernelArgPtr);
    END_TRY_R("creating HalfArg", NULL)
}

jfloatArray Java_yacx_HalfArg_asFloatArray (JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
    	CHECK_NULL(kernelArgJNIPtr, NULL)
        auto data = kernelArgJNIPtr->getHostData();
        auto dataSize = kernelArgJNIPtr->kernelArgPtr()->size();

        auto res = env->NewFloatArray(dataSize / sizeof(jfloat));

        CHECK_NULL(res, NULL)

        void* dataFloat = malloc(dataSize*2);
        convertHtoF(data, dataFloat, dataSize);

        env->SetFloatArrayRegion(res, 0, dataSize / sizeof(jfloat),
                                   reinterpret_cast<const jfloat*>(dataFloat));

        free(dataFloat);

        return res;
    END_TRY_R("getting FloatArg-content", NULL)
}

jobject Java_yacx_HalfArg_asFloatArg(JNIEnv* env, jobject obj){
    BEGIN_TRY
        auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, obj);
    	CHECK_NULL(kernelArgJNIPtr, NULL)
        auto kernelArgPtr = kernelArgJNIPtr->kernelArgPtr();
        auto data = kernelArgJNIPtr->getHostData();
        auto dataSize = kernelArgPtr->size();

        jclass cls = getClass(env, "yacx/FloatArg");
		if (cls == NULL) return NULL;

        KernelArgJNI* newkernelArgJNIPtr = new KernelArgJNI{NULL, dataSize*2,
            kernelArgPtr->isDownload(), kernelArgPtr->isCopy(), true};

        convertHtoF(data, kernelArgJNIPtr->getHostData(), dataSize);

		return createJNIObject(env, cls, newkernelArgJNIPtr);
    END_TRY_R("converting HalfArg to FloatArg", NULL)
}

