#include "outputImageSeq.hpp"

using namespace cv;

int main()
{
	Mat image = imread("C:/Users/jpeop/dissertation/csc3002_image_dehazing/csc3002_new/forest.jpg", 1);
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

// Copy data to img
memcpy(img, floatImage.data, sizeof(float) * width * height * channels);

// Allocate memory for darkChannelImg
float* darkChannelImg = (float*)calloc(width * height, sizeof(float));
if (darkChannelImg == NULL) {
    printf("Memory allocation for darkChannelImg failed\n");
    free(img); // Free previously allocated memory
    return 1;
}

// Allocate memory for transmissionImg
float* transmissionImg = (float*)calloc(width * height, sizeof(float));
if (transmissionImg == NULL) {
    printf("Memory allocation for transmissionImg failed\n");
    free(img); // Free previously allocated memory
    free(darkChannelImg); // Free previously allocated memory
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
	for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; j++) {
        r[i * width + j] = img[(i * width + j) * channels];
        g[i * width + j] = img[(i * width + j) * channels + 1];
        b[i * width + j] = img[(i * width + j) * channels + 2];
    }
}

		// Create buffers for RGB channels
		cl::Buffer rBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * r.size(), r.data());
		cl::Buffer gBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * g.size(), g.data());
		cl::Buffer bBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * b.size(), b.data());

		cl::Buffer imageBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * width * height * channels, img);
		cl::Buffer darkChannelBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * width * height);
		cl::Buffer atmosphereBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3);
		cl::Buffer transEstBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * width * height);
		cl::Buffer radianceBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * width * height * 3);

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
		get_dark_channel.setArg(6, 16);
		get_dark_channel.setArg(7, darkChannelBuffer);

		queue.enqueueNDRangeKernel(get_dark_channel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NullRange);
		queue.finish();

		queue.enqueueReadBuffer(darkChannelBuffer, CL_TRUE, 0, sizeof(float) * width * height, darkChannelImg);

		std::cout << "Dark Channel Image (First 10 elements):" << std::endl;
for (int i = 0; i < 10; ++i) {
    std::cout << darkChannelImg[i] << " ";
}
std::cout << std::endl;

		get_atmosphere.setArg(0, imageBuffer);
		get_atmosphere.setArg(1, atmosphereBuffer);
		get_atmosphere.setArg(2, cl::Local(1024 * sizeof(float)));
		get_atmosphere.setArg(3, width * height);

		queue.enqueueNDRangeKernel(get_atmosphere, cl::NullRange, cl::NDRange(width * height), cl::NullRange);
		queue.finish();

		queue.enqueueReadBuffer(atmosphereBuffer, CL_TRUE, 0, sizeof(float) * 3, atmosphere);

		std::cout << "Atmosphere: " << atmosphere[0] << ", " << atmosphere[1] << ", " << atmosphere[2] << std::endl;

		get_transmission_estimate.setArg(0, imageBuffer);
		get_transmission_estimate.setArg(1, atmosphereBuffer);
		get_transmission_estimate.setArg(2, transEstBuffer);
		get_transmission_estimate.setArg(3, 0.90f);
		get_transmission_estimate.setArg(4, height);
		get_transmission_estimate.setArg(5, width);

		queue.enqueueNDRangeKernel(get_transmission_estimate, cl::NullRange, cl::NDRange(width * height), cl::NullRange);
		queue.finish();

       queue.enqueueReadBuffer(transEstBuffer, CL_TRUE, 0, sizeof(float) * width * height, transmissionImg);

	   std::cout << "Transmission Estimate (First 10 elements):" << std::endl;
for (int i = 0; i < 10; ++i) {
    std::cout << transmissionImg[i] << " ";
}
std::cout << std::endl;

		get_radiance.setArg(0, imageBuffer);
		get_radiance.setArg(1, transEstBuffer);
		get_radiance.setArg(2, atmosphereBuffer);
		get_radiance.setArg(3, radianceBuffer);
		get_radiance.setArg(4, width);
        get_radiance.setArg(5, height);
		
		queue.enqueueNDRangeKernel(get_radiance, cl::NullRange, cl::NDRange(width * height), cl::NullRange);
		queue.finish();

		std::vector<float> result(width * height * 3); 
        queue.enqueueReadBuffer(radianceBuffer, CL_TRUE, 0, sizeof(float) * result.size(), result.data());

		std::cout << "Radiance (First 10 elements):" << std::endl;
for (int i = 0; i < 10; ++i) {
    std::cout << result[i] << " ";
}
std::cout << std::endl;

		queue.finish();

		auto stop = std::chrono::high_resolution_clock::now();
	    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;

        Mat imgcv_out(height, width, CV_32FC3, result.data()); 
        imwrite("C:/Users/jpeop/dissertation/csc3002_image_dehazing/csc3002_new/sequentialresult.png", imgcv_out);

		cv::Mat darkChannelMat(height, width, CV_32FC1, darkChannelImg);
        cv::Mat transmissionMat(height, width, CV_32FC1, transmissionImg);
		cv::normalize(darkChannelMat, darkChannelMat, 0, 255, cv::NORM_MINMAX);
        cv::normalize(transmissionMat, transmissionMat, 0, 255, cv::NORM_MINMAX);
		darkChannelMat.convertTo(darkChannelMat, CV_8UC1);
        transmissionMat.convertTo(transmissionMat, CV_8UC1);
		cv::imwrite("dark_channel.png", darkChannelMat);
        cv::imwrite("transmission.png", transmissionMat);

	}
	catch (cl::Error err)
	{
		std::cerr << " Exception: "
				  << err.what()
				  << err.err()
				  << std::endl;
	}

	free(img);
    free(darkChannelImg);
    free(transmissionImg);

	std::cout << "Press ENTER to exit...";
	std::cin.get();

	return 0;
}