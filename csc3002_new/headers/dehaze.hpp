#ifndef DEHAZE_HPP
#define DEHAZE_HPP

#include <CL/opencl.hpp>

void dehaze(cl::Context& context, cl::CommandQueue& queue, cl::Buffer& imageBuffer, float omega, int win_size, cl::Buffer& darkChannelBuffer, cl::Buffer& atmosphereBuffer, cl::Buffer& transEstBuffer, cl::Buffer& radianceBuffer, int m, int n);

#endif

