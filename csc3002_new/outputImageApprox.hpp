#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <tuple>
#include <string>
#include <chrono>
// #include "matplotlibcpp.h"
#include <opencv2/opencv.hpp>
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define CL_VERSION_1_2
#define __CL_ENABLE_EXCEPTIONS

 #if defined(_WIN32) || defined(_WIN64)
 #include <CL/opencl.hpp>
 #else
 #include <CL/opencl.hpp>
 #endif