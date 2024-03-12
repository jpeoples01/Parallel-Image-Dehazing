#define CL_HPP_MINIMUM_OPENCL_VERSION BASE_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>
#include <iostream>

int main() {
   std::vector<cl::Platform> all_platforms;
   cl::Platform::get(&all_platforms);

   for (auto& platform : all_platforms) {
       std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";

       std::vector<cl::Device> all_devices;
       platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);

       for (auto& device : all_devices) {
           std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";
       }
   }

std::cout << "Press ENTER to exit...";
    std::cin.get();
   return 0;
}