# Minimal version of CMake
cmake_minimum_required (VERSION 3.5)
 
# Define project name
project (OpenCL_Dehaze)

# Find OpenCV
set(OpenCV_DIR "C:/Users/jpeop/Downloads/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("C:/Users/jpeop/OpenCL-CLHPP/include")
include_directories("C:/Users/jpeop/openCL-Headers")
include_directories("C:/Users/jpeop/OpenCL-SDK")

# Compiler flags
if (CMAKE_COMPILER_IS_GNUCC)
    set (CMAKE_CXX_FLAGS "-D_REETRANT -Wall -Wextra -pedantic -Wno-long-long")
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
   	    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0")
	elseif( CMAKE_BUILD_TYPE STREQUAL "Release" )
	    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -fno-strict-aliasing")
	endif ()
endif (CMAKE_COMPILER_IS_GNUCC)
 
# Set up executable for outputImageApprox.cpp
add_executable (imagedehazeApprox outputImageApprox.cpp)
target_link_libraries(imagedehazeApprox "C:/Users/jpeop/OpenCL-SDK/install/lib/OpenCL.lib")
target_link_libraries(imagedehazeApprox ${OpenCV_LIBS})

# Set up executable for outputImageSeq.cpp
add_executable (imagedehazeSeq outputImageSeq.cpp)
target_link_libraries(imagedehazeSeq "C:/Users/jpeop/OpenCL-SDK/install/lib/OpenCL.lib")
target_link_libraries(imagedehazeSeq ${OpenCV_LIBS})

# Set up executable for outputImageSeq.cpp
add_executable (imagedehazeHalf outputImageHalf.cpp)
target_link_libraries(imagedehazeHalf "C:/Users/jpeop/OpenCL-SDK/install/lib/OpenCL.lib")
target_link_libraries(imagedehazeHalf ${OpenCV_LIBS})
