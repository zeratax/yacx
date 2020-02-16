#include "yacx_Executor.h"
#include "Handle.h"
#include "KernelUtils.h"
#include "KernelArgJNI.hpp"
#include "../../include/yacx/KernelTime.hpp"
#include "../../include/yacx/Kernel.hpp"
#include "../../include/yacx/Devices.hpp"

#include <cstring>
#include <stdio.h>

using yacx::Kernel, yacx::Device;

jobjectArray Java_yacx_Executor_benchmark (JNIEnv* env, jclass cls, jobject jkernel, jobject jdevice,
		jobjectArray jArgs, jint jexecutions){
	BEGIN_TRY
		CHECK_BIGGER(jexecutions, 0, "illegal number of executions", NULL)
		CHECK_NULL(jkernel, NULL);
		CHECK_NULL(jdevice, NULL);
		CHECK_NULL(jArgs, NULL);

		auto kernelPtr = getHandle<Kernel>(env, jkernel);
		CHECK_NULL(kernelPtr, NULL);
		auto devicePtr = getHandle<Device>(env, jdevice);
		CHECK_NULL(devicePtr, NULL);

		auto args = getArguments(env, jArgs);
		if (args.empty()) return NULL;

		jclass kernelTimeCls = getClass(env, "yacx/KernelTime");
		if (!kernelTimeCls) return NULL;

		//Run benchmark-test
		auto kernelTimes = kernelPtr->benchmark(args, jexecutions, devicePtr);

		//Create Output-Array
		auto res = (jobjectArray) env->NewObjectArray(jexecutions, kernelTimeCls, NULL);
		CHECK_NULL(res, NULL)

		for (unsigned int i = 0; i < jexecutions; ++i){
			auto jkernelTime = createJavaKernelTime(env, &kernelTimes.at(i));
			env->SetObjectArrayElement(res, i, jkernelTime);
		}

		return res;
	END_TRY_R("benchmarking Kernel", NULL)
}
