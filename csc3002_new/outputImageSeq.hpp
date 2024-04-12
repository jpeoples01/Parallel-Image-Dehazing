#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <tuple>
#include <string>
#include <chrono>
#include <algorithm>
// #include "matplotlibcpp.h"
#include <opencv2/opencv.hpp>

#define CL_VERSION_1_2
#define __CL_ENABLE_EXCEPTIONS

 #if defined(_WIN32) || defined(_WIN64)
 #include <CL/opencl.hpp>
 #else
 #endif