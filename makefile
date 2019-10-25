CC := g++ # This is the main compiler
# CC := clang --analyze # and comment out the linker last line for sanity
LINTER := cpplint
SRCDIR := src
BUILDDIR := build
TARGET := bin/runner
OUT_DIR := bin kernel lib src test
 
SRCEXT := cpp
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
CFLAGS := -Wall -g
LIB := -lboost_program_options # -lnvrtc -lcuda -L $(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 
INC := # -I include -I $(CUDA_PATH)/include

# Build
all: directories $(TARGET)

$(TARGET): $(OBJECTS)
	@echo " Linking..."
	@echo " $(CC) $^ -o $(TARGET) $(LIB)"; $(CC) $^ -o $(TARGET) $(LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	@echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CFLAGS) $(INC) -c -o $@ $<

MKDIR_P = mkdir -p

directories: ${OUT_DIR}

${OUT_DIR}:
	${MKDIR_P} ${OUT_DIR}

clean:
	@echo " Cleaning..."; 
	@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; $(RM) -r $(BUILDDIR) $(TARGET)

# Tests
tester:
	$(CC) $(CFLAGS) test/tester.cpp $(INC) $(LIB) -o bin/tester

# Linter
lint:
	$(LINTER) --root=${CURDIR} --recursive . 

.PHONY: clean, lint, tester