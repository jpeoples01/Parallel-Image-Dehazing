#ifndef TRANSMISSION_ESTIMATE_HPP
#define TRANSMISSION_ESTIMATE_HPP

#include <CL/opencl.hpp>

void get_transmission_estimate(cl::Context& context, cl::CommandQueue& queue, cl::Buffer& imageBuffer, cl::Buffer& atmosphereBuffer, float omega, int win_size, cl::Buffer& darkChannelBuffer, cl::Buffer& transEstBuffer, int m, int n);

#endif
