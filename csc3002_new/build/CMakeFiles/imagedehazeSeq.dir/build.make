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
CMAKE_SOURCE_DIR = /home/jetsonnano2g/csc3002_image_dehazing/csc3002_new

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jetsonnano2g/csc3002_image_dehazing/csc3002_new/build

# Include any dependencies generated for this target.
include CMakeFiles/imagedehazeSeq.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/imagedehazeSeq.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/imagedehazeSeq.dir/flags.make

CMakeFiles/imagedehazeSeq.dir/outputImageSeq.cpp.o: CMakeFiles/imagedehazeSeq.dir/flags.make
CMakeFiles/imagedehazeSeq.dir/outputImageSeq.cpp.o: ../outputImageSeq.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jetsonnano2g/csc3002_image_dehazing/csc3002_new/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/imagedehazeSeq.dir/outputImageSeq.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/imagedehazeSeq.dir/outputImageSeq.cpp.o -c /home/jetsonnano2g/csc3002_image_dehazing/csc3002_new/outputImageSeq.cpp

CMakeFiles/imagedehazeSeq.dir/outputImageSeq.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/imagedehazeSeq.dir/outputImageSeq.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jetsonnano2g/csc3002_image_dehazing/csc3002_new/outputImageSeq.cpp > CMakeFiles/imagedehazeSeq.dir/outputImageSeq.cpp.i

CMakeFiles/imagedehazeSeq.dir/outputImageSeq.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/imagedehazeSeq.dir/outputImageSeq.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jetsonnano2g/csc3002_image_dehazing/csc3002_new/outputImageSeq.cpp -o CMakeFiles/imagedehazeSeq.dir/outputImageSeq.cpp.s

CMakeFiles/imagedehazeSeq.dir/outputImageSeq.cpp.o.requires:

.PHONY : CMakeFiles/imagedehazeSeq.dir/outputImageSeq.cpp.o.requires

CMakeFiles/imagedehazeSeq.dir/outputImageSeq.cpp.o.provides: CMakeFiles/imagedehazeSeq.dir/outputImageSeq.cpp.o.requires
	$(MAKE) -f CMakeFiles/imagedehazeSeq.dir/build.make CMakeFiles/imagedehazeSeq.dir/outputImageSeq.cpp.o.provides.build
.PHONY : CMakeFiles/imagedehazeSeq.dir/outputImageSeq.cpp.o.provides

CMakeFiles/imagedehazeSeq.dir/outputImageSeq.cpp.o.provides.build: CMakeFiles/imagedehazeSeq.dir/outputImageSeq.cpp.o


# Object files for target imagedehazeSeq
imagedehazeSeq_OBJECTS = \
"CMakeFiles/imagedehazeSeq.dir/outputImageSeq.cpp.o"

# External object files for target imagedehazeSeq
imagedehazeSeq_EXTERNAL_OBJECTS =

imagedehazeSeq: CMakeFiles/imagedehazeSeq.dir/outputImageSeq.cpp.o
imagedehazeSeq: CMakeFiles/imagedehazeSeq.dir/build.make
imagedehazeSeq: /usr/lib/aarch64-linux-gnu/libOpenCL.so
imagedehazeSeq: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.1.1
imagedehazeSeq: /usr/lib/aarch64-linux-gnu/libopencv_gapi.so.4.1.1
imagedehazeSeq: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.1.1
imagedehazeSeq: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.1.1
imagedehazeSeq: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.1.1
imagedehazeSeq: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.1.1
imagedehazeSeq: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.1.1
imagedehazeSeq: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.1.1
imagedehazeSeq: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.1.1
imagedehazeSeq: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.1.1
imagedehazeSeq: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.1.1
imagedehazeSeq: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.1.1
imagedehazeSeq: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.1.1
imagedehazeSeq: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.1.1
imagedehazeSeq: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.1.1
imagedehazeSeq: CMakeFiles/imagedehazeSeq.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jetsonnano2g/csc3002_image_dehazing/csc3002_new/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable imagedehazeSeq"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/imagedehazeSeq.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/imagedehazeSeq.dir/build: imagedehazeSeq

.PHONY : CMakeFiles/imagedehazeSeq.dir/build

CMakeFiles/imagedehazeSeq.dir/requires: CMakeFiles/imagedehazeSeq.dir/outputImageSeq.cpp.o.requires

.PHONY : CMakeFiles/imagedehazeSeq.dir/requires

CMakeFiles/imagedehazeSeq.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/imagedehazeSeq.dir/cmake_clean.cmake
.PHONY : CMakeFiles/imagedehazeSeq.dir/clean

CMakeFiles/imagedehazeSeq.dir/depend:
	cd /home/jetsonnano2g/csc3002_image_dehazing/csc3002_new/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jetsonnano2g/csc3002_image_dehazing/csc3002_new /home/jetsonnano2g/csc3002_image_dehazing/csc3002_new /home/jetsonnano2g/csc3002_image_dehazing/csc3002_new/build /home/jetsonnano2g/csc3002_image_dehazing/csc3002_new/build /home/jetsonnano2g/csc3002_image_dehazing/csc3002_new/build/CMakeFiles/imagedehazeSeq.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/imagedehazeSeq.dir/depend

