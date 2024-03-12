#ifndef MIN_KERNEL_HPP
#define MIN_KERNEL_HPP

#include <CL/opencl.hpp>

void minKernel(cl::Context& context, cl::CommandQueue& queue, cl::Buffer& rBuffer, cl::Buffer& gBuffer, cl::Buffer& bBuffer, int x_height, int x_width, int wnd, cl::Buffer& darkChannelBuffer, cl::Buffer& d_infoBuffer, int startThreadNum);

#endif