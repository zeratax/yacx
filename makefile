CC := g++ # This is the main compiler
# CC := clang --analyze # and comment out the linker last line for sanity
LINTER := cpplint
FORMATER := clang-format
MKDIR_P = mkdir -p

SRCDIR := src
LIBDIR := include
BUILDDIR := build
EXAMPLES := examples
TARGET := bin/runner
OUT_DIR := bin build
 
SRCEXT := cpp
HEADDEREXT := hpp
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
HEADERS := $(shell find $(LIBDIR) -type f -name *.$(HEADDEREXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o)) $(BUILDDIR)/example.o
CFLAGS := -std=c++17 -Wall -g -DNVRTC_GET_TYPE_NAME=1
LIB := -lnvrtc -lcuda -L $(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 
INC := -I $(LIBDIR) -I $(CUDA_PATH)/include

# Build
example1: EXAMPLE := $(EXAMPLES)/example1.cpp
example1: default
default: directories buildexample $(TARGET)

directories: ${OUT_DIR}
buildexample: $(EXAMPLE)
	@echo " Building $(EXAMPLE)";
	@echo " $(CC) $(CFLAGS) $(INC) -c -o $(BUILDDIR)/example.o $(EXAMPLE)"; $(CC) $(CFLAGS) $(INC) -c -o $(BUILDDIR)/example.o $(EXAMPLE)

${OUT_DIR}:
	${MKDIR_P} ${OUT_DIR}

$(TARGET): $(OBJECTS)
	@echo " Linking...";
	@echo " $(CC) $^ -o $(TARGET) $(LIB)"; $(CC) $^ -o $(TARGET) $(LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CFLAGS) $(INC) -c -o $@ $<

clean:
	@echo " Cleaning..."; 
	@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; $(RM) -r $(BUILDDIR) $(TARGET)

# Format
format:
	$(FORMATER) -i -style=file $(SOURCES) $(HEADERS)

# Linter
lint:
	$(LINTER) --root=${CURDIR} --recursive .
	#clang-tidy src/ -system-headers=false

# Tests
tester:
	$(CC) $(CFLAGS) test/tester.cpp $(INC) $(LIB) -o bin/tester

.PHONY: clean, lint, directories, all, format