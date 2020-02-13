#include "CExecutor.hpp"

#include <iostream>
#include <sstream>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/stat.h> 
#include <fcntl.h>
#include <bits/stdc++.h> 

// using yacx::KernelArg;

struct opfn{ void (*op)(void** parameter);};

struct dynop{
	void (*op)(void** parameter);
	void *libhandle;
};

void load_op(struct dynop *dest, const char *filename) {
    //open libary
    void* handle = dlopen((std::string("./") + filename).c_str(), RTLD_LAZY);
    char* error = dlerror();
    if (handle == NULL){
        if (error != NULL){
            throw std::runtime_error(std::string("error while open libary with compiled function") + error);
        } else {
            throw std::runtime_error("error while open libary with compiled function");
        }
    }

    //Search op-struct in libary
    void* op = dlsym(handle, "op");
    error = dlerror();
    if (op == NULL){
        dlclose(handle);
        
        if (error != NULL){
            throw std::runtime_error(std::string("error while search \"op\" in libary with"
            "compiled function") + error);
        } else {
            throw std::runtime_error("error while search \"op\" in libary with compiled function");
        }
    }

    //Save result in passed struct
    dest->op = ((struct opfn*) op)->op;
    dest->libhandle = handle;
}

void unload_op(struct dynop *op) {
    if (op == NULL){
        return;
    }

    //Close libary
    int result = dlclose(op->libhandle);
    char* error = dlerror();
    if (result != 0){
        errno = EIO;
        if (error != NULL){
            perror((std::string("error while close libary with compiled function") + error).c_str());
        } else {
            perror("error while close libary with compiled function");
        }
    }

    //clear struct
    op->libhandle = NULL;
    op->op = NULL;
}

// void execute(CProgram* cProgram, std::vector<KernelArg> kernelArgs) {
void execute(CProgram* cProgram, std::vector<void*>& kernelArgs) {
    struct dynop op;

	//open libary
	load_op(&op, cProgram->getLibFile());

	//execute function
    void** parameter = &kernelArgs[0];
    op.op(parameter);

	//close libary
	unload_op(&op);
}