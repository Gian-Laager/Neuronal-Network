# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

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
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/gianlaager/Desktop/Neuronal Network"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/gianlaager/Desktop/Neuronal Network/cmake-build-debug"

# Include any dependencies generated for this target.
include CMakeFiles/Neuronal_Network.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Neuronal_Network.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Neuronal_Network.dir/flags.make

CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx.pch: CMakeFiles/Neuronal_Network.dir/flags.make
CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx.pch: CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx.cxx
CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx.pch: CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/gianlaager/Desktop/Neuronal Network/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx.pch"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Xclang -emit-pch -Xclang -include -Xclang "/Users/gianlaager/Desktop/Neuronal Network/cmake-build-debug/CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx" -o CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx.pch -c "/Users/gianlaager/Desktop/Neuronal Network/cmake-build-debug/CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx.cxx"

CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Xclang -emit-pch -Xclang -include -Xclang "/Users/gianlaager/Desktop/Neuronal Network/cmake-build-debug/CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx" -E "/Users/gianlaager/Desktop/Neuronal Network/cmake-build-debug/CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx.cxx" > CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx.i

CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Xclang -emit-pch -Xclang -include -Xclang "/Users/gianlaager/Desktop/Neuronal Network/cmake-build-debug/CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx" -S "/Users/gianlaager/Desktop/Neuronal Network/cmake-build-debug/CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx.cxx" -o CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx.s

CMakeFiles/Neuronal_Network.dir/src/Neural_Network.cpp.o: CMakeFiles/Neuronal_Network.dir/flags.make
CMakeFiles/Neuronal_Network.dir/src/Neural_Network.cpp.o: ../src/Neural_Network.cpp
CMakeFiles/Neuronal_Network.dir/src/Neural_Network.cpp.o: CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx
CMakeFiles/Neuronal_Network.dir/src/Neural_Network.cpp.o: CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx.pch
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/gianlaager/Desktop/Neuronal Network/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Neuronal_Network.dir/src/Neural_Network.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Xclang -include-pch -Xclang "/Users/gianlaager/Desktop/Neuronal Network/cmake-build-debug/CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx.pch" -Xclang -include -Xclang "/Users/gianlaager/Desktop/Neuronal Network/cmake-build-debug/CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx" -o CMakeFiles/Neuronal_Network.dir/src/Neural_Network.cpp.o -c "/Users/gianlaager/Desktop/Neuronal Network/src/Neural_Network.cpp"

CMakeFiles/Neuronal_Network.dir/src/Neural_Network.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Neuronal_Network.dir/src/Neural_Network.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Xclang -include-pch -Xclang "/Users/gianlaager/Desktop/Neuronal Network/cmake-build-debug/CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx.pch" -Xclang -include -Xclang "/Users/gianlaager/Desktop/Neuronal Network/cmake-build-debug/CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx" -E "/Users/gianlaager/Desktop/Neuronal Network/src/Neural_Network.cpp" > CMakeFiles/Neuronal_Network.dir/src/Neural_Network.cpp.i

CMakeFiles/Neuronal_Network.dir/src/Neural_Network.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Neuronal_Network.dir/src/Neural_Network.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Xclang -include-pch -Xclang "/Users/gianlaager/Desktop/Neuronal Network/cmake-build-debug/CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx.pch" -Xclang -include -Xclang "/Users/gianlaager/Desktop/Neuronal Network/cmake-build-debug/CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx" -S "/Users/gianlaager/Desktop/Neuronal Network/src/Neural_Network.cpp" -o CMakeFiles/Neuronal_Network.dir/src/Neural_Network.cpp.s

# Object files for target Neuronal_Network
Neuronal_Network_OBJECTS = \
"CMakeFiles/Neuronal_Network.dir/src/Neural_Network.cpp.o"

# External object files for target Neuronal_Network
Neuronal_Network_EXTERNAL_OBJECTS =

libNeuronal_Network.a: CMakeFiles/Neuronal_Network.dir/cmake_pch.hxx.pch
libNeuronal_Network.a: CMakeFiles/Neuronal_Network.dir/src/Neural_Network.cpp.o
libNeuronal_Network.a: CMakeFiles/Neuronal_Network.dir/build.make
libNeuronal_Network.a: CMakeFiles/Neuronal_Network.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/gianlaager/Desktop/Neuronal Network/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libNeuronal_Network.a"
	$(CMAKE_COMMAND) -P CMakeFiles/Neuronal_Network.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Neuronal_Network.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Neuronal_Network.dir/build: libNeuronal_Network.a

.PHONY : CMakeFiles/Neuronal_Network.dir/build

CMakeFiles/Neuronal_Network.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Neuronal_Network.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Neuronal_Network.dir/clean

CMakeFiles/Neuronal_Network.dir/depend:
	cd "/Users/gianlaager/Desktop/Neuronal Network/cmake-build-debug" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/gianlaager/Desktop/Neuronal Network" "/Users/gianlaager/Desktop/Neuronal Network" "/Users/gianlaager/Desktop/Neuronal Network/cmake-build-debug" "/Users/gianlaager/Desktop/Neuronal Network/cmake-build-debug" "/Users/gianlaager/Desktop/Neuronal Network/cmake-build-debug/CMakeFiles/Neuronal_Network.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/Neuronal_Network.dir/depend

