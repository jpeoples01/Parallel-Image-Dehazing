#ifndef ATMOSPHERE_HPP
#define ATMOSPHERE_HPP

#include <CL/opencl.hpp>

void get_atmosphere(cl::Context& context, cl::CommandQueue& queue, cl::Buffer& imageBuffer, cl::Buffer& darkChannelBuffer, cl::Buffer& accumulatorBuffer, int m, int n);

#endif

