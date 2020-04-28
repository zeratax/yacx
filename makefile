ANALYZER := clang --analyze
LINTER := cpplint
FORMATER := clang-format
MKDIR_P = mkdir -p

SRCDIR := src
INCDIR := include
VENDOR := extern
TESTDIR := test
BUILDDIR := build
EXAMPLEDIR := examples
DIRS := ${BUILDDIR}

SRCEXT := cpp
HEADDEREXT := hpp
SOURCES := $(shell find $(SRCDIR) -maxdepth 1 -type f -name '*.$(SRCEXT)')
TESTS := $(shell find $(TESTDIR) -type f -name '*.$(SRCEXT)')
EXAMPLES := $(shell find $(EXAMPLEDIR) -maxdepth 1 -type f -name '*.$(SRCEXT)')
HEADERS := $(shell find $(INCDIR) -type f -name '*.$(HEADDEREXT)')

directories: ${DIRS}

${DIRS}:
	${MKDIR_P} ${DIRS}

clean:
	@echo " Cleaning..."; 
	@echo " $(RM) -r $(DIRS) docs/{java,html,latex}"; $(RM) -r $(DIRS) docs/{java,html,latex}

format:
	$(FORMATER) -i -style=file $(SOURCES) $(HEADERS) $(TESTS) $(EXAMPLES)

lint:
	$(LINTER) --root=${CURDIR} --recursive .
	#clang-tidy src/ -system-headers=false

docs:
	doxygen Doxyfile
	javadoc -Xdoclint:none -d docs/java -sourcepath src/main/java yacx

.PHONY: clean, lint, directories, format, docs

