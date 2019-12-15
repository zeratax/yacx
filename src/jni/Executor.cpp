#include "Executor.h"
#include "Handle.h"
#include "Kernel.h"
#include "KernelUtils.h"
#include "KernelArgJNI.hpp"
#include "../../include/yacx/KernelTime.hpp"
#include "../../include/yacx/Kernel.hpp"
#include "../../include/yacx/Device.hpp"

#include <cstring>
#include <stdio.h>

using yacx::Kernel, yacx::Device;

jobjectArray Java_Executor_benchmark (JNIEnv* env, jclass cls, jobject jkernel, jobject jdevice,
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

		jclass kernelTimeCls = getClass(env, "KernelTime");
		if (!kernelTimeCls) return NULL;

		//Create Output-Array
		auto res = (jobjectArray) env->NewObjectArray(jexecutions, kernelTimeCls, NULL);
		CHECK_NULL(res, NULL)

		//Backup host-data
		void* hostData[args.size()];
		void* hostDataBackup[args.size()];
		for(size_t i = 0; i != args.size(); i++) {
			auto jkernelArg = env->GetObjectArrayElement(jArgs, i);
			auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, jkernelArg);

			size_t size = kernelArgJNIPtr->kernelArgPtr()->size();
			void* data = kernelArgJNIPtr->getHostData();

			hostData[i] = data;
			hostDataBackup[i] = malloc(size);
			std::memcpy(hostDataBackup[i], data, size);
		}

		//Run first n-1 executions
		for (int i = 0; i < jexecutions-1; i++) {
			auto jkerneltime = Java_Kernel_launchInternal__LDevice_2_3LKernelArg_2(env, jkernel, jdevice, jArgs);

			if (jkerneltime == NULL) return NULL;

			env->SetObjectArrayElement(res, i, jkerneltime);

			//Restore host-data
			for(size_t i = 0; i != args.size(); i++) {
				std::memcpy(hostData[i], hostDataBackup[i], args[i].size());
			}
		}

		//Free backuped host-data
		for(size_t i = 0; i != args.size(); i++) {
			free(hostDataBackup[i]);
		}

		//Last run without restore Hostdata
		auto jkerneltime = Java_Kernel_launchInternal__LDevice_2_3LKernelArg_2(env, jkernel, jdevice, jArgs);

		if (jkerneltime == NULL) return NULL;

		env->SetObjectArrayElement(res, jexecutions-1, jkerneltime);

		return res;
	END_TRY("benchmarking Kernel")
}
