#define CL_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <iostream>
#include <vector>
#include <CL/opencl.h>
#include <CL/opencl.hpp>

int main() {
    const int vectorSize = 10;

    std::vector<int> A(vectorSize, 1);
    std::vector<int> B(vectorSize, 2);
    std::vector<int> C(vectorSize, 0);

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    err = clGetPlatformIDs(1, &platform, nullptr);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    const char* source = "__kernel void add(__global const int* A, __global const int* B, __global int* C) { \
                            int i = get_global_id(0); \
                            C[i] = A[i] + B[i]; \
                          }";

    program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    kernel = clCreateKernel(program, "add", &err);

    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(int) * vectorSize, A.data(), &err);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(int) * vectorSize, B.data(), &err);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(int) * vectorSize, nullptr, &err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);

    size_t globalSize = vectorSize;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    err = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, sizeof(int) * vectorSize, C.data(), 0, nullptr, nullptr);

    for (int i = 0; i < vectorSize; ++i) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    std::cin.get();
    return 0;
}