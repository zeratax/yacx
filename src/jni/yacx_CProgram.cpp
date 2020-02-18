#include "yacx_CProgram.h"
#include "Handle.h"
#include "KernelArgJNI.hpp"
#include "KernelUtils.h"
#include "../../include/yacx/KernelArgs.hpp"
#include "../../include/yacx/Options.hpp"
#include "../../include/yacx/cexecutor/CProgram.hpp"

#include <iostream>
#include <sstream>

using yacx::KernelArg, jni::KernelArgJNI, yacx::Options, yacx::CProgram;

jobject Java_yacx_CProgram_createInternal(JNIEnv* env, jclass cls, jstring jcProgram, jstring jcFunctionName,
  jint jnumberParameter, jstring jcompiler, jobject joptions){
    BEGIN_TRY
        CHECK_NULL(jcProgram, NULL)
        CHECK_NULL(jcFunctionName, NULL)
        CHECK_BIGGER(jnumberParameter, 0, "illegal number of parameter", NULL)
        CHECK_NULL(jcompiler, NULL)
        CHECK_NULL(joptions, NULL)

        auto cProgram = env->GetStringUTFChars(jcProgram, nullptr);
        auto cFunctionName = env->GetStringUTFChars(jcFunctionName, nullptr);
        auto compiler = env->GetStringUTFChars(jcompiler, nullptr);

        auto optionsPtr = getHandle<Options>(env, joptions);
        CHECK_NULL(optionsPtr, NULL);

        std::stringstream compilerWithOptionsS;
        compilerWithOptionsS << compiler << " ";
        for (int i = 0; i < optionsPtr->numOptions(); i++){
          compilerWithOptionsS << optionsPtr->content()[i] << " ";
        }
        std::string tmp = compilerWithOptionsS.str();
        auto compilerWithOptions = tmp.c_str();

        CProgram* cProgramPtr = new CProgram{cProgram, cFunctionName, jnumberParameter, compilerWithOptions};

        return createJNIObject(env, cls, cProgramPtr);
    END_TRY_R("creating cProgram", NULL)
}

void Java_yacx_CProgram_execute(JNIEnv* env, jobject obj, jobjectArray jKernelArgs){
    BEGIN_TRY
        CHECK_NULL(jKernelArgs, )

        auto cProgramPtr = getHandle<CProgram>(env, obj);
        CHECK_NULL(cProgramPtr, );

        auto argumentsLength = env->GetArrayLength(jKernelArgs);
        void* argsPtr[argumentsLength];

        CHECK_BIGGER(argumentsLength, 0, "illegal array length", )
        CHECK_BIGGER(argumentsLength, cProgramPtr->getNumberArguments()-1, "illegal number of arguments", )
        CHECK_BIGGER(cProgramPtr->getNumberArguments(), argumentsLength-1, "illegal number of arguments", )

        for(int i = 0; i < argumentsLength; i++){
            auto jkernelArg = env->GetObjectArrayElement(jKernelArgs, i);
            CHECK_NULL(jkernelArg, )

            auto kernelArgJNIPtr = getHandle<KernelArgJNI>(env, jkernelArg);
            CHECK_NULL(kernelArgJNIPtr, )

            argsPtr[i] = kernelArgJNIPtr->getHostData();
        }

        //cProgramPtr->execute(argsPtr);
    END_TRY("executing cProgram")
}
