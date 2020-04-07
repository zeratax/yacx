#include "../../include/yacx/cexecutor/LibaryLoader.hpp"
#include "../../include/yacx/Logger.hpp"

#include <iostream>
#include <sstream>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/stat.h> 
#include <fcntl.h>

using yacx::loglevel, yacx::detail::dynop, yacx::detail::opfn;

void yacx::detail::load_op(struct dynop *dest, std::string filename, std::string opSymbolName) {
    //open libary
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
}

void yacx::detail::unload_op(struct dynop *op) {
    if (op == NULL){
        return;
    }

    //Close libary
    int result = dlclose(op->libhandle);
    char* error = dlerror();
    if (result != 0){
        if (error != NULL){
            Logger(loglevel::ERROR) << "error while closing libary with compiled function: " << error;
        } else {
            Logger(loglevel::ERROR) << "unknown error while closing libary with compiled function";
        }
    }

    //clear struct
    op->libhandle = NULL;
    op->op = NULL;
}