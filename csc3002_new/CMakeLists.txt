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

# Source code of application		
set (opencl_src outputImage.cpp stb_image.cpp stb_image_write.cpp)
 
# Compiler flags
if (CMAKE_COMPILER_IS_GNUCC)
    set (CMAKE_CXX_FLAGS "-D_REETRANT -Wall -Wextra -pedantic -Wno-long-long")
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
   	    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0")
	elseif( CMAKE_BUILD_TYPE STREQUAL "Release" )
	    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -fno-strict-aliasing")
	endif ()
endif (CMAKE_COMPILER_IS_GNUCC)
 
# Set up executable
add_executable (imagedehaze ${opencl_src})
target_link_libraries(imagedehaze "C:/Users/jpeop/OpenCL-SDK/install/lib/OpenCL.lib")
target_link_libraries(imagedehaze ${OpenCV_LIBS})



