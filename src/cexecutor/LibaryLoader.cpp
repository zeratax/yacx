#include "../../include/yacx/cexecutor/LibaryLoader.hpp"
#include "../../include/yacx/Logger.hpp"

#include <iostream>
#include <sstream>
#if _MSC_VER
#else
#include <dlfcn.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/stat.h> 
#include <fcntl.h>

using yacx::loglevel, yacx::detail::dynop, yacx::detail::opfn;

void yacx::detail::load_op(struct dynop *dest, std::string filename, std::string opSymbolName) {
    //open libary
    #ifndef _MSC_VER
    void* handle = dlopen((std::string("./") + filename).c_str(), RTLD_LAZY);
    char* error = dlerror();
    if (handle == NULL){
        if (error != NULL){
            throw std::runtime_error(std::string("error while opening libary with compiled function ") + error);
        } else {
            throw std::runtime_error("error while opening libary with compiled function");
        }
    }

    //Search op-struct in libary
    void* op = dlsym(handle, opSymbolName.c_str());
    error = dlerror();
    if (op == NULL){
        dlclose(handle);
        
        if (error != NULL){
            throw std::runtime_error(std::string("error while searching\"") + std::string(opSymbolName) +
            std::string("\" in libary with compiled function ") + error);
        } else {
            throw std::runtime_error(std::string("error while searching\"") + std::string(opSymbolName) +
            std::string("\" in libary with compiled function "));
        }
    }

    //Save result in passed struct
    dest->op = ((struct opfn*) op)->op;
    dest->libhandle = handle;
    #endif
}

void yacx::detail::unload_op(struct dynop *op) {
    #ifndef _MSC_VER
    if (op == NULL){
        return;
    }

    //Close libary
    int result = dlclose(op->libhandle);
    char* error = dlerror();
    if (result != 0){
        if (error != NULL){
            Logger(loglevel::ERR) << "error while closing libary with compiled function: " << error;
        } else {
            Logger(loglevel::ERR) << "unknown error while closing libary with compiled function";
        }
    }

    //clear struct
    op->libhandle = NULL;
    op->op = NULL;
    #endif
}