CC := g++ # This is the main compiler
# CC := clang --analyze # and comment out the linker last line for sanity

SRCDIR := .
INCDIR := ../include
VENDOR := ../extern
BUILDDIR := ../build
TARGET := ../bin/tester
 
SRCEXT := cpp
HEADDEREXT := hpp
SOURCES := $(shell find $(SRCDIR) -type f -name '*.$(SRCEXT)')
HEADERS := $(shell find $(LIBDIR) -type f -name '*.$(HEADDEREXT)')
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
CFLAGS := -std=c++17 -Wall -g -DNVRTC_GET_TYPE_NAME=1
LIB := -lnvrtc -lcuda -L $(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 
INC := -I $(INCDIR) -I $(CUDA_PATH)/include -I $(VENDOR)/catch2/single_include

# Build
all: $(OBJECTS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CFLAGS) $(INC) -c -o $@ $<

.PHONY: all