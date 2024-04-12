# Minimal version of CMake
cmake_minimum_required (VERSION 3.5)

# Define project name
project (OpenCL_Dehaze)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find OpenCL (PoCL should provide this)
find_package(OpenCL REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCL_INCLUDE_DIRS})
include_directories("/home/jetsonnano2g/pocl/include")

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
target_link_libraries(imagedehazeApprox ${OpenCL_LIBRARIES})
target_link_libraries(imagedehazeApprox ${OpenCV_LIBS})

# Set up executable for outputImageSeq.cpp
add_executable (imagedehazeSeq outputImageSeq.cpp)
target_link_libraries(imagedehazeSeq ${OpenCL_LIBRARIES})
target_link_libraries(imagedehazeSeq ${OpenCV_LIBS})


# Set up executable for outputImageHalf.cpp
add_executable (imagedehazeHalf outputImageHalf.cpp)
target_link_libraries(imagedehazeHalf ${OpenCL_LIBRARIES})
target_link_libraries(imagedehazeHalf ${OpenCV_LIBS})