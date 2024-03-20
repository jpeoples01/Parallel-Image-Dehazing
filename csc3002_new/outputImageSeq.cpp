#include "outputImageSeq.hpp"

using namespace cv;

int main()
{
	Mat image = imread("C:/Users/jpeop/dissertation/csc3002_image_dehazing/csc3002_new/forest.jpg", 0);
	int width = image.size().width;
	int height = image.size().height;
	int channels = image.channels();

	cv::Mat floatImage;
    image.convertTo(floatImage, CV_32F);

	float* img = (float*)malloc(sizeof(float) * width * height * channels);
	if (img == NULL) {
		printf("Memory allocation for img failed\n");
		return 1;
	}

	memcpy(img, floatImage.data, sizeof(float) * width * height * channels);

	float* img_out = (float*)malloc(sizeof(float) * width * height * channels);
	if (img_out == NULL) {
		printf("Memory allocation for img_out failed\n");
		free(img); 
		return 1;
	}

	float atmosphere[3];
	for (int i = 0; i < 10; ++i)
	{ // Print the first 10 pixels
		std::cout << "Pixel " << i << ": ";
		for (int c = 0; c < channels; ++c)
		{
			std::cout << img[i * channels + c] << " ";
		}
		std::cout << "\n";
	}

	// size_t localWorkSize = 1024;
	size_t globalWorkSize = (width * height);

	try
	{
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
		platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
		device = devices[0];
		// Create context
		context = cl::Context(device);
		// Create command queue
		queue = cl::CommandQueue(context, device);
		// cl::Context context(CL_DEVICE_TYPE_GPU);
		// cl::CommandQueue queue(context);
		// cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>().front();

		size_t maxWorkGroupSize;
		clGetDeviceInfo(device(), CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);

		// Split image into RGB channels
		std::vector<float> r(width * height);
		std::vector<float> g(width * height);
		std::vector<float> b(width * height);
		for (int i = 0; i < height; ++i)
		{ for (int j = 0; j < width; j++) {
			r[i] = img[(i * width + j) * channels];
			g[i] = img[(i * width + j) * channels + 1];
			b[i] = img[(i * width + j) * channels + 2];
		} }

		// Create buffers for RGB channels
		cl::Buffer rBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * r.size(), r.data());
		cl::Buffer gBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * g.size(), g.data());
		cl::Buffer bBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * b.size(), b.data());

		cl::Buffer imageBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * width * height * channels, img);
		cl::Buffer darkChannelBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * width * height * channels);
		cl::Buffer atmosphereBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3);
		cl::Buffer transEstBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * width * height * channels);
		cl::Buffer radianceBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * width * height * channels);

		// Create and build programs
		cl::Program::Sources sources;

		// Load the OpenCL source code
		cl_int err;

		std::ifstream file("../dehazeSeq.cl");
		std::string source(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
		sources.push_back({source.c_str(), source.length() + 1});

		// Compile the OpenCL source code
		cl::Program program(context, sources);
		program.build("-cl-std=CL1.2");

		cl::Kernel get_dark_channel(program, "get_dark_channel");
		cl::Kernel get_atmosphere(program, "get_atmosphere");
		cl::Kernel get_transmission_estimate(program, "get_transmission_estimate");
		cl::Kernel get_radiance(program, "get_radiance");

		auto start = std::chrono::high_resolution_clock::now();

		queue.enqueueWriteBuffer(imageBuffer, CL_TRUE, 0, sizeof(float) * width * height * channels, img);
		std::cout << "Wrote to image buffer" << std::endl;

		get_dark_channel.setArg(0, 0);
		get_dark_channel.setArg(1, rBuffer);
		get_dark_channel.setArg(2, gBuffer);
		get_dark_channel.setArg(3, bBuffer);
		get_dark_channel.setArg(4, height);
		get_dark_channel.setArg(5, width);
		get_dark_channel.setArg(6, 15);
		get_dark_channel.setArg(7, darkChannelBuffer);

		queue.enqueueNDRangeKernel(get_dark_channel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NullRange);
		queue.finish();

		queue.enqueueReadBuffer(darkChannelBuffer, CL_TRUE, 0, sizeof(float) * width * height * channels, img_out);

		err = get_atmosphere.setArg(0, imageBuffer);
		if (err != CL_SUCCESS)
		{
			std::cerr << "Error setting kernel argument 0: " << err << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Arguement 0 successful" << std::endl;
		}
		err = get_atmosphere.setArg(1, atmosphereBuffer);
		if (err != CL_SUCCESS)
		{
			std::cerr << "Error setting kernel argument 1: " << err << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Arguement 1 successful" << std::endl;
		}
		get_atmosphere.setArg(2, cl::Local(1024 * sizeof(float)));
		if (err != CL_SUCCESS)
		{
			std::cerr << "Error setting kernel argument 2: " << err << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Arguement 2 successful" << std::endl;
		}
		get_atmosphere.setArg(3, width * height);
		if (err != CL_SUCCESS)
		{
			std::cerr << "Error setting kernel argument 3: " << err << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Arguement 3 successful" << std::endl;
		}

		cl_int ret3 = queue.enqueueNDRangeKernel(get_atmosphere, cl::NullRange, cl::NDRange(width * height), cl::NullRange);
		if (ret3 != CL_SUCCESS)
		{
			std::cerr << "Failed to execute Atmosphere kernel: " << ret3 << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Kernel execution successful" << std::endl;
		}
		ret3 = queue.finish();
		if (ret3 != CL_SUCCESS)
		{
			std::cerr << "Failed to finish command queue: " << ret3 << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Command queue finished" << std::endl;
		}

		queue.enqueueReadBuffer(atmosphereBuffer, CL_TRUE, 0, sizeof(float) * 3, atmosphere);

		get_transmission_estimate.setArg(0, imageBuffer);
		get_transmission_estimate.setArg(1, atmosphereBuffer);
		get_transmission_estimate.setArg(2, 0.80f);
		get_transmission_estimate.setArg(3, 15);
		get_transmission_estimate.setArg(4, darkChannelBuffer);
		get_transmission_estimate.setArg(5, transEstBuffer);
		get_transmission_estimate.setArg(6, height);
		get_transmission_estimate.setArg(7, width);
		cl_int ret4 = queue.enqueueNDRangeKernel(get_transmission_estimate, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
		if (ret4 != CL_SUCCESS)
		{
			std::cerr << "Failed to execute Transmission kernel: " << ret4 << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Kernel execution successful" << std::endl;
		}
		ret4 = queue.finish();
		if (ret4 != CL_SUCCESS)
		{
			std::cerr << "Failed in command queue: " << ret4 << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Command queue finished" << std::endl;
		}

       queue.enqueueReadBuffer(transEstBuffer, CL_TRUE, 0, sizeof(float) * width * height * channels, img_out);

		get_radiance.setArg(0, imageBuffer);
		get_radiance.setArg(1, transEstBuffer);
		get_radiance.setArg(2, atmosphereBuffer);
		get_radiance.setArg(3, radianceBuffer);
		get_radiance.setArg(4, height);
		get_radiance.setArg(5, width);
		cl_int ret5 = queue.enqueueNDRangeKernel(get_radiance, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
		if (ret5 != CL_SUCCESS)
		{
			std::cerr << "Failed to execute Radiance kernel: " << ret5 << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Kernel Execution Successful" << std::endl;
		}
		cl_int ret6 = queue.finish();
		if (ret6 != CL_SUCCESS)
		{
			std::cerr << "Failed to finish command queue: " << ret6 << std::endl;
			return 1;
		}
		else
		{
			std::cout << "Command queue finished" << std::endl;
		}

		std::vector<float> result(width * height * channels); 
err = queue.enqueueReadBuffer(radianceBuffer, CL_TRUE, 0, sizeof(float) * result.size(), result.data());
		if (err != CL_SUCCESS)
		{
			std::cerr << "Error reading Radiance buffer: " << err << std::endl;
		}

		for (int i = 0; i < 1000; ++i)
		{
			std::cout << "Element " << i << ": " << result[i] << "\n";
		}

		queue.finish();

		auto stop = std::chrono::high_resolution_clock::now();
	    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;

        Mat imgcv_out(height, width, CV_32FC1, result.data()); 
        imwrite("C:/Users/jpeop/dissertation/csc3002_image_dehazing/csc3002_new/sequentialresult.png", imgcv_out);
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