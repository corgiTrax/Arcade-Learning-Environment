# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ruohan/ale

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ruohan/ale

# Include any dependencies generated for this target.
include CMakeFiles/fifoInterfaceExample.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fifoInterfaceExample.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fifoInterfaceExample.dir/flags.make

CMakeFiles/fifoInterfaceExample.dir/doc/examples/fifoInterfaceExample.cpp.o: CMakeFiles/fifoInterfaceExample.dir/flags.make
CMakeFiles/fifoInterfaceExample.dir/doc/examples/fifoInterfaceExample.cpp.o: doc/examples/fifoInterfaceExample.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ruohan/ale/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/fifoInterfaceExample.dir/doc/examples/fifoInterfaceExample.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/fifoInterfaceExample.dir/doc/examples/fifoInterfaceExample.cpp.o -c /home/ruohan/ale/doc/examples/fifoInterfaceExample.cpp

CMakeFiles/fifoInterfaceExample.dir/doc/examples/fifoInterfaceExample.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fifoInterfaceExample.dir/doc/examples/fifoInterfaceExample.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ruohan/ale/doc/examples/fifoInterfaceExample.cpp > CMakeFiles/fifoInterfaceExample.dir/doc/examples/fifoInterfaceExample.cpp.i

CMakeFiles/fifoInterfaceExample.dir/doc/examples/fifoInterfaceExample.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fifoInterfaceExample.dir/doc/examples/fifoInterfaceExample.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ruohan/ale/doc/examples/fifoInterfaceExample.cpp -o CMakeFiles/fifoInterfaceExample.dir/doc/examples/fifoInterfaceExample.cpp.s

CMakeFiles/fifoInterfaceExample.dir/doc/examples/fifoInterfaceExample.cpp.o.requires:
.PHONY : CMakeFiles/fifoInterfaceExample.dir/doc/examples/fifoInterfaceExample.cpp.o.requires

CMakeFiles/fifoInterfaceExample.dir/doc/examples/fifoInterfaceExample.cpp.o.provides: CMakeFiles/fifoInterfaceExample.dir/doc/examples/fifoInterfaceExample.cpp.o.requires
	$(MAKE) -f CMakeFiles/fifoInterfaceExample.dir/build.make CMakeFiles/fifoInterfaceExample.dir/doc/examples/fifoInterfaceExample.cpp.o.provides.build
.PHONY : CMakeFiles/fifoInterfaceExample.dir/doc/examples/fifoInterfaceExample.cpp.o.provides

CMakeFiles/fifoInterfaceExample.dir/doc/examples/fifoInterfaceExample.cpp.o.provides.build: CMakeFiles/fifoInterfaceExample.dir/doc/examples/fifoInterfaceExample.cpp.o

# Object files for target fifoInterfaceExample
fifoInterfaceExample_OBJECTS = \
"CMakeFiles/fifoInterfaceExample.dir/doc/examples/fifoInterfaceExample.cpp.o"

# External object files for target fifoInterfaceExample
fifoInterfaceExample_EXTERNAL_OBJECTS =

doc/examples/ale-fifoInterfaceExample: CMakeFiles/fifoInterfaceExample.dir/doc/examples/fifoInterfaceExample.cpp.o
doc/examples/ale-fifoInterfaceExample: CMakeFiles/fifoInterfaceExample.dir/build.make
doc/examples/ale-fifoInterfaceExample: /usr/lib/x86_64-linux-gnu/libSDLmain.a
doc/examples/ale-fifoInterfaceExample: /usr/lib/x86_64-linux-gnu/libSDL.so
doc/examples/ale-fifoInterfaceExample: CMakeFiles/fifoInterfaceExample.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable doc/examples/ale-fifoInterfaceExample"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fifoInterfaceExample.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fifoInterfaceExample.dir/build: doc/examples/ale-fifoInterfaceExample
.PHONY : CMakeFiles/fifoInterfaceExample.dir/build

CMakeFiles/fifoInterfaceExample.dir/requires: CMakeFiles/fifoInterfaceExample.dir/doc/examples/fifoInterfaceExample.cpp.o.requires
.PHONY : CMakeFiles/fifoInterfaceExample.dir/requires

CMakeFiles/fifoInterfaceExample.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fifoInterfaceExample.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fifoInterfaceExample.dir/clean

CMakeFiles/fifoInterfaceExample.dir/depend:
	cd /home/ruohan/ale && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ruohan/ale /home/ruohan/ale /home/ruohan/ale /home/ruohan/ale /home/ruohan/ale/CMakeFiles/fifoInterfaceExample.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fifoInterfaceExample.dir/depend

