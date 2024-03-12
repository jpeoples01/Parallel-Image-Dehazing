#include "outputImage.hpp"

using namespace cv;

int main()
{
	Mat image = imread("C:/Users/jpeop/csc3002_new/forest.jpg", 0);
int width = image.size().width;
int height = image.size().height;
int channels = image.channels();

cv::Mat int16Image;
image.convertTo(int16Image, CV_16S); // Convert the image to int16

int16_t* img = (int16_t*)malloc(sizeof(int16_t) * width * height * channels);
if (img == NULL) {
    printf("Memory allocation for img failed\n");
    return 1;
}

memcpy(img, int16Image.data, sizeof(int16_t) * width * height * channels);

int16_t* img_out = (int16_t*)malloc(sizeof(int16_t) * width * height * channels);
if (img_out == NULL) {
    printf("Memory allocation for img_out failed\n");
    free(img); 
    return 1;
}

int16_t atmosphere[3];
for (int i = 0; i < 10; ++i) { // Print the first 10 pixels
    std::cout << "Pixel " << i << ": ";
    for (int c = 0; c < channels; ++c) {
        std::cout << img[i * channels + c] << " ";
    }
    std::cout << "\n";
}

	std::cout << "Image read successfully. Width: " << width << ", Height: " << height << ", Channels: " << channels << std::endl;

	// size_t localWorkSize = 1024;
	size_t globalWorkSize = (width * height);

	try
	{
		std::cout << "Initializing OpenCL..." << std::endl;
		cl::Platform platform;
		cl::Device device;
		cl::Context context;
		cl::CommandQueue queue;
		// Get platform
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		platform = platforms[1];
		// Get device
		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
		device = devices[0];
		// Create context
		context = cl::Context(device);
		// Create command queue
		queue = cl::CommandQueue(context, device);
		// cl::Context context(CL_DEVICE_TYPE_GPU);
		// cl::CommandQueue queue(context);
		// cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>().front();
		std::cout << "Done initializing OpenCL." << std::endl;

		size_t maxWorkGroupSize;
		clGetDeviceInfo(device(), CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
		std::cout << "Max Work Group Size: " << maxWorkGroupSize << std::endl;

		// Split image into RGB channels
		std::vector<int16_t> r(width * height);
		std::vector<int16_t> g(width * height);
		std::vector<int16_t> b(width * height);
		for (int i = 0; i < height; ++i)
		{ for (int j = 0; j < width; j++) {
			r[i] = img[(i * width + j) * channels];
			g[i] = img[(i * width + j) * channels + 1];
			b[i] = img[(i * width + j) * channels + 2];
		} }

		// Create buffers for RGB channels
		cl::Buffer rBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int16_t) * r.size(), r.data());
		cl::Buffer gBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int16_t) * g.size(), g.data());
		cl::Buffer bBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int16_t) * b.size(), b.data());

		cl::Buffer imageBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int16_t) * width * height * channels, img);
		std::cout << "Created image buffer" << std::endl;
		cl::Buffer darkChannelBuffer(context, CL_MEM_READ_WRITE, sizeof(int16_t) * width * height * channels);
		std::cout << "Created dark channel buffer" << std::endl;
		cl::Buffer atmosphereBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3);
		std::cout << "Created atmosphere buffer" << std::endl;
		cl::Buffer transEstBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * width * height * channels);
		std::cout << "Created transmission buffer" << std::endl;
		cl::Buffer radianceBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * width * height * channels);
		std::cout << "Created radiance buffer" << std::endl;

		// Create and build programs
		std::cout << "Creating and building programs..." << std::endl;
		cl::Program::Sources sources;

		// Load the OpenCL source code
		cl_int err;

		std::ifstream file("../dehaze.cl");
		std::string source(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
		sources.push_back({source.c_str(), source.length() + 1});

		// Compile the OpenCL source code
		cl::Program program(context, sources);
		program.build("-cl-std=CL1.2");

        // cl::Kernel dehaze(program, "dehaze");
		cl::Kernel get_dark_channel(program, "get_dark_channel");
		// cl::Kernel get_atmosphere(program, "get_atmosphere");
		// cl::Kernel get_transmission_estimate(program, "get_transmission_estimate");
		// cl::Kernel get_radiance(program, "get_radiance");

		queue.enqueueWriteBuffer(imageBuffer, CL_TRUE, 0, sizeof(int16_t) * width * height * channels, img);
		std::cout << "Wrote to image buffer" << std::endl;

		err = get_dark_channel.setArg(0, 0);
		if (err != CL_SUCCESS)
		{
			std::cerr << "Error setting kernel argument 0: " << err << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Arguement 0 successful" << std::endl;
		}
		err = get_dark_channel.setArg(1, rBuffer);
		if (err != CL_SUCCESS)
		{
			std::cerr << "Error setting kernel argument 1: " << err << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Arguement 1 successful" << std::endl;
		}
		err = get_dark_channel.setArg(2, gBuffer);
		if (err != CL_SUCCESS)
		{
			std::cerr << "Error setting kernel argument 2: " << err << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Arguement 2 successful" << std::endl;
		}
		err = get_dark_channel.setArg(3, bBuffer);
		if (err != CL_SUCCESS)
		{
			std::cerr << "Error setting kernel argument 3: " << err << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Arguement 3 successful" << std::endl;
		}
		err = get_dark_channel.setArg(4, height);
		if (err != CL_SUCCESS)
		{
			std::cerr << "Error setting kernel argument 4: " << err << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Arguement 4 successful" << std::endl;
		}
		err = get_dark_channel.setArg(5, width);
		if (err != CL_SUCCESS)
		{
			std::cerr << "Error setting kernel argument 5: " << err << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Arguement 5 successful" << std::endl;
		}
		err = get_dark_channel.setArg(6, 15);
		if (err != CL_SUCCESS)
		{
			std::cerr << "Error setting kernel argument 6: " << err << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Arguement 6 successful" << std::endl;
		}
		err = get_dark_channel.setArg(7, darkChannelBuffer);
		if (err != CL_SUCCESS)
		{
			std::cerr << "Error setting kernel argument 7: " << err << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Arguement 7 successful" << std::endl;
		}

		cl_int ret2 = queue.enqueueNDRangeKernel(get_dark_channel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NullRange);
		if (ret2 != CL_SUCCESS)
		{
			std::cerr << "Failed to execute Dark Channel kernel: " << ret2 << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Kernel executed successfully" << std::endl;
		}
		ret2 = queue.finish();
		if (ret2 != CL_SUCCESS)
		{
			std::cerr << "Failed to finish command queue: " << ret2 << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Command queue finished" << std::endl;
		}

	// 	queue.enqueueReadBuffer(darkChannelBuffer, CL_TRUE, 0, sizeof(float) * width * height * channels, img_out);

	// 	err = get_atmosphere.setArg(0, imageBuffer);
	// 	if (err != CL_SUCCESS)
	// 	{
	// 		std::cerr << "Error setting kernel argument 0: " << err << std::endl;
	// 		return 1;
	// 	}
	// 	else
	// 	{
	// 		std::cout << "Arguement 0 successful" << std::endl;
	// 	}
	// 	err = get_atmosphere.setArg(1, atmosphereBuffer);
	// 	if (err != CL_SUCCESS)
	// 	{
	// 		std::cerr << "Error setting kernel argument 1: " << err << std::endl;
	// 		return 1;
	// 	}
	// 	else
	// 	{
	// 		std::cout << "Arguement 1 successful" << std::endl;
	// 	}
	// 	get_atmosphere.setArg(2, cl::Local(localWorkSize * sizeof(float)));
	// 	if (err != CL_SUCCESS)
	// 	{
	// 		std::cerr << "Error setting kernel argument 2: " << err << std::endl;
	// 		return 1;
	// 	}
	// 	else
	// 	{
	// 		std::cout << "Arguement 2 successful" << std::endl;
	// 	}
	// 	get_atmosphere.setArg(3, width * height);
	// 	if (err != CL_SUCCESS)
	// 	{
	// 		std::cerr << "Error setting kernel argument 3: " << err << std::endl;
	// 		return 1;
	// 	}
	// 	else
	// 	{
	// 		std::cout << "Arguement 3 successful" << std::endl;
	// 	}

	// 	cl_int ret3 = queue.enqueueNDRangeKernel(get_atmosphere, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
	// 	if (ret3 != CL_SUCCESS)
	// 	{
	// 		std::cerr << "Failed to execute Atmosphere kernel: " << ret3 << std::endl;
	// 		return 1;
	// 	}
	// 	else
	// 	{
	// 		std::cout << "Kernel execution successful" << std::endl;
	// 	}
	// 	ret3 = queue.finish();
	// 	if (ret3 != CL_SUCCESS)
	// 	{
	// 		std::cerr << "Failed to finish command queue: " << ret3 << std::endl;
	// 		return 1;
	// 	}
	// 	else
	// 	{
	// 		std::cout << "Command queue finished" << std::endl;
	// 	}

	// 	queue.enqueueReadBuffer(atmosphereBuffer, CL_TRUE, 0, sizeof(float) * 3, atmosphere);

	// 	get_transmission_estimate.setArg(0, imageBuffer);
	// 	get_transmission_estimate.setArg(1, atmosphereBuffer);
	// 	get_transmission_estimate.setArg(2, 0.80f);
	// 	get_transmission_estimate.setArg(3, 15);
	// 	get_transmission_estimate.setArg(4, darkChannelBuffer);
	// 	get_transmission_estimate.setArg(5, transEstBuffer);
	// 	get_transmission_estimate.setArg(6, height);
	// 	get_transmission_estimate.setArg(7, width);
	// 	cl_int ret4 = queue.enqueueNDRangeKernel(get_transmission_estimate, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
	// 	if (ret4 != CL_SUCCESS)
	// 	{
	// 		std::cerr << "Failed to execute Transmission kernel: " << ret4 << std::endl;
	// 		return 1;
	// 	}
	// 	else
	// 	{
	// 		std::cout << "Kernel execution successful" << std::endl;
	// 	}
	// 	ret4 = queue.finish();
	// 	if (ret4 != CL_SUCCESS)
	// 	{
	// 		std::cerr << "Failed in command queue: " << ret4 << std::endl;
	// 		return 1;
	// 	}
	// 	else
	// 	{
	// 		std::cout << "Command queue finished" << std::endl;
	// 	}

	// 	queue.enqueueReadBuffer(transEstBuffer, CL_TRUE, 0, sizeof(float) * height * width, img);

	// 	get_radiance.setArg(0, imageBuffer);
	// 	get_radiance.setArg(1, transEstBuffer);
	// 	get_radiance.setArg(2, atmosphereBuffer);
	// 	get_radiance.setArg(3, radianceBuffer);
	// 	get_radiance.setArg(4, height);
	// 	get_radiance.setArg(5, width);
	// 	cl_int ret5 = queue.enqueueNDRangeKernel(get_radiance, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
	// 	if (ret5 != CL_SUCCESS)
	// 	{
	// 		std::cerr << "Failed to execute Radiance kernel: " << ret5 << std::endl;
	// 		return 1;
	// 	}
	// 	else
	// 	{
	// 		std::cout << "Kernel Execution Successful" << std::endl;
	// 	}
	// 	cl_int ret6 = queue.finish();
	// 	if (ret6 != CL_SUCCESS)
	// 	{
	// 		std::cerr << "Failed to finish command queue: " << ret6 << std::endl;
	// 		return 1;
	// 	}
	// 	else
	// 	{
	// 		std::cout << "Command queue finished" << std::endl;
	// 	}

    //  // Set the kernel arguments
	// 	err = dehaze.setArg(0, imageBuffer);
    //   if (err != CL_SUCCESS) {
    //   std::cerr << "Error setting kernel argument 0: " << err << std::endl;}
	// 	err = dehaze.setArg(1, darkChannelBuffer);
    //   if (err != CL_SUCCESS) {
    //   std::cerr << "Error setting kernel argument 1: " << err << std::endl;}
	// 	err = dehaze.setArg(2, atmosphereBuffer);
    //   if (err != CL_SUCCESS) {
    //   std::cerr << "Error setting kernel argument 2: " << err << std::endl;}
	// 	err = dehaze.setArg(3, transEstBuffer);
    //   if (err != CL_SUCCESS) {
    //   std::cerr << "Error setting kernel argument 3: " << err << std::endl;}
	// 	err = dehaze.setArg(4, radianceBuffer);
    //   if (err != CL_SUCCESS) {
    //   std::cerr << "Error setting kernel argument 4: " << err << std::endl;}
	// 	err = dehaze.setArg(5, width);
    //   if (err != CL_SUCCESS) {
    //   std::cerr << "Error setting kernel argument 5: " << err << std::endl;}
	// 	err = dehaze.setArg(6, height);
    //   if (err != CL_SUCCESS) {
    //   std::cerr << "Error setting kernel argument 6: " << err << std::endl;}
	// 	err = dehaze.setArg(7, 0.80f);  // Set the omega parameter
    //   if (err != CL_SUCCESS) {
    //   std::cerr << "Error setting kernel argument 7: " << err << std::endl;}
	// 	err = dehaze.setArg(8, 15);  // Set the window size parameter
    //   if (err != CL_SUCCESS) {
    //   std::cerr << "Error setting kernel argument 8: " << err << std::endl;}

	// 	// Enqueue the kernel for execution
	// 	queue.enqueueNDRangeKernel(dehaze, cl::NullRange, cl::NDRange(width, height), cl::NullRange);

	// 	queue.finish();
		
		// std::vector<float> result(width * height * channels);
		std::vector<int16_t> result(width * height);
		err = queue.enqueueReadBuffer(darkChannelBuffer, CL_TRUE, 0, sizeof(int16_t) * result.size(), result.data());
		if (err != CL_SUCCESS)
		{
			std::cerr << "Error reading Radiance buffer: " << err << std::endl;
		}

		for (int i = 0; i < 100; ++i)
		{
			std::cout << "Element " << i << ": " << result[i] << "\n";
		}

		queue.finish();

		std::cout << "Done executing kernels." << std::endl;

		std::cout << "About to write image..." << std::endl;
		// write_image("result.png", result, width, height, channels);
		Mat  imgcv_out(height, width, CV_16SC1, img);
		imwrite("result.png", imgcv_out);
		std::cout << "Image written successfully..." << std::endl;
	}
	catch (cl::Error err)
	{
		std::cerr << " Exception: "
				  << err.what()
				  << err.err()
				  << std::endl;
	}

	std::cout << "Press ENTER to exit...";
	std::cin.get();

	return 0;
}