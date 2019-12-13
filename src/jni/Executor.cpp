#include "Executor.h"
#include "Handle.h"
#include "Kernel.h"
#include "../../include/cudaexecutor/KernelTime.hpp"

jobjectArray Java_Executor_benchmark (JNIEnv* env, jclass cls, jobject jkernel, jobject jdevice,
		jobjectArray jArgs, jint jexecutions){
	BEGIN_TRY
		CHECK_BIGGER(jexecutions, 0, "illegal number of executions", NULL)

		jclass kernelTimeCls = getClass(env, "KernelTime");

		if (!kernelTimeCls) return NULL;

		auto res = (jobjectArray) env->NewObjectArray(jexecutions, kernelTimeCls, NULL);
		CHECK_NULL(res, NULL)

		for (int i = 0; i < jexecutions; i++){
			auto jkerneltime = Java_Kernel_launchInternal__LDevice_2_3LKernelArg_2(env, jkernel, jdevice, jArgs);

			env->SetObjectArrayElement(res, i, jkerneltime);
		}

		return res;
	END_TRY("benchmarking Kernel")
}
