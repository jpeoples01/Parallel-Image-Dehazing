#ifndef RADIANCE_HPP
#define RADIANCE_HPP

#include <CL/opencl.hpp>

void get_radiance(cl::Context& context, cl::CommandQueue& queue, cl::Buffer& imageBuffer, cl::Buffer& transmissionBuffer, cl::Buffer& atmosphereBuffer, cl::Buffer& radianceBuffer, int m, int n);

#endif
