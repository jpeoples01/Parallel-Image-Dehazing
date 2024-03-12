#ifndef DEHAZE_HPP
#define DEHAZE_HPP


#define CL_VERSION_1_2
#define __CL_ENABLE_EXCEPTIONS

 #if defined(_WIN32) || defined(_WIN64)
 #include <CL/opencl.hpp>
 #else
 #include <CL/opencl.hpp>
 #endif

void dehaze(cl::Context& context, cl::CommandQueue& queue, cl::Buffer& imageBuffer, float omega, int win_size, cl::Buffer& darkChannelBuffer, cl::Buffer& atmosphereBuffer, cl::Buffer& transEstBuffer, cl::Buffer& radianceBuffer, int m, int n);

#endif