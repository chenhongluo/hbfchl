# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_SOURCE_DIR = /home/chl/hbftest/HBF/HBF

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chl/hbftest/HBF/HBF/build

# Utility rule file for clean-all.

# Include the progress variables for this target.
include CMakeFiles/clean-all.dir/progress.make

CMakeFiles/clean-all:
	rm -rf *

clean-all: CMakeFiles/clean-all
clean-all: CMakeFiles/clean-all.dir/build.make

.PHONY : clean-all

# Rule to build all files generated by this target.
CMakeFiles/clean-all.dir/build: clean-all

.PHONY : CMakeFiles/clean-all.dir/build

CMakeFiles/clean-all.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/clean-all.dir/cmake_clean.cmake
.PHONY : CMakeFiles/clean-all.dir/clean

CMakeFiles/clean-all.dir/depend:
	cd /home/chl/hbftest/HBF/HBF/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chl/hbftest/HBF/HBF /home/chl/hbftest/HBF/HBF /home/chl/hbftest/HBF/HBF/build /home/chl/hbftest/HBF/HBF/build /home/chl/hbftest/HBF/HBF/build/CMakeFiles/clean-all.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/clean-all.dir/depend

