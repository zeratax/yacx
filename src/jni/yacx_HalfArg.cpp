#include "yacx_HalfArg.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "Half.hpp"
#include "../../include/yacx/KernelArgs.hpp"

using yacx::KernelArg, jni::KernelArgJNI, yacx::convertFtoH, yacx::convertHtoF;

jobject JNICALL Java_yacx_HalfArg_createValue(JNIEnv* env, jclass cls, jfloat jvalue){
	BEGIN_TRY
		cls = getClass(env, "yacx/KernelArg");
		if (cls == NULL) return NULL;

		KernelArgJNI* kernelArgPtr = new KernelArgJNI{NULL, sizeof(jfloat)/2, false, false, false};
        convertFtoH(&jvalue, kernelArgPtr->getHostData(), 1);

		return createJNIObject(env, cls, kernelArgPtr);
	END_TRY_R("creating HalfValueArg", NULL)
}

jobject Java_yacx_HalfArg_createInternal (JNIEnv* env, jclass cls, jfloatArray jarray, jboolean jdownload){
    BEGIN_TRY
        CHECK_NULL(jarray, NULL)

        auto arrayPtr = env->GetFloatArrayElements(jarray, NULL);
        auto arrayLength = env->GetArrayLength(jarray);

        CHECK_BIGGER(arrayLength, 0, "illegal array length", NULL)

        KernelArgJNI* kernelArgPtr = new KernelArgJNI{NULL, arrayLength * (sizeof(jfloat)/2), jdownload, true, true};
        convertFtoH(arrayPtr, kernelArgPtr->getHostData(), arrayLength);

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

        auto res = env->NewFloatArray(dataSize*2 / sizeof(jfloat));

        CHECK_NULL(res, NULL)

        void* dataFloatTmp = malloc(dataSize*2);
        convertHtoF(data, dataFloatTmp, dataSize/2);

        env->SetFloatArrayRegion(res, 0, dataSize*2 / sizeof(jfloat),
                                   reinterpret_cast<const jfloat*>(dataFloatTmp));

        free(dataFloatTmp);

        return res;
    END_TRY_R("getting HalfArg-content", NULL)
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

        convertHtoF(data, newkernelArgJNIPtr->getHostData(), dataSize/2);

		return createJNIObject(env, cls, newkernelArgJNIPtr);
    END_TRY_R("converting HalfArg to FloatArg", NULL)
}

