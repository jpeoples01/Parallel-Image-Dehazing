#define CL_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <iostream>
#include <vector>
#include <CL/opencl.h>
#include <CL/opencl.hpp>

int main() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        return 1;
    }

    cl::Platform defaultPlatform = platforms[0];
    std::vector<cl::Device> devices;
    defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    if (devices.empty()) {
        std::cerr << "No OpenCL devices found." << std::endl;
        return 1;
    }

    cl::Device defaultDevice = devices[0];
    cl::Context context(defaultDevice);
    cl::Program::Sources sources;
    sources.push_back({"__kernel void hello() { printf(\"Hello, World!\\n\"); }"});
    cl::Program program(context, sources);

    if (program.build({defaultDevice}) != CL_SUCCESS) {
        std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDevice) << std::endl;
        return 1;
    }

    cl::Kernel kernel(program, "hello");

    // Create an output buffer to hold the printf output from the kernel
    const int OUTPUT_BUFFER_SIZE = 1024; // Adjust this size based on your expected output
    char outputBuffer[OUTPUT_BUFFER_SIZE];
    cl::Buffer clOutputBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(outputBuffer), outputBuffer);

    // Set the output buffer as an argument to the kernel
    kernel.setArg(0, clOutputBuffer);

    cl::CommandQueue queue(context, defaultDevice);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NullRange);

    // Read the output buffer back to the host
    queue.enqueueReadBuffer(clOutputBuffer, CL_TRUE, 0, sizeof(outputBuffer), outputBuffer);

    // Print the output buffer content
    std::cout << "Kernel Output: " << std::endl;
    std::cout << outputBuffer << std::endl;

    std::cin.get();
    return 0;
}
