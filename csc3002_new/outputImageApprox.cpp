#include "outputImageApprox.hpp"

using namespace cv;

int main()
{
	Mat image = imread("C:/Users/jpeop/dissertation/csc3002_image_dehazing/csc3002_new/forest.jpg", cv::IMREAD_COLOR);
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

	int width = image.size().width;
	int height = image.size().height;
	int channels = image.channels();

	cv::Mat halfImage;
	image.convertTo(halfImage, CV_16FC1, 1.0 / 255.0);

	std::vector<cl_half> img(width * height * channels);
	memcpy(img.data(), halfImage.data, sizeof(cl_half) * width * height * channels);

	std::vector<cl_half> darkChannelImg(width * height);
	std::vector<cl_half> transmissionImg(width * height);
	cl_half atmosphere[3];

	for (int i = 0; i < 10; ++i)
	{
		std::cout << "Pixel " << i << ": ";
		for (int c = 0; c < channels; ++c)
		{
			std::cout << img[i * channels + c] << " ";
		}
		std::cout << "\n";
	}

	size_t globalWorkSize = (width * height);

	try
	{
		cl::Platform platform;
		cl::Device device;
		cl::Context context;
		cl::CommandQueue queue;

		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		platform = platforms[1];

		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
		device = devices[0];

		std::string extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
		if (extensions.find("cl_khr_fp16") != std::string::npos)
		{
			std::cout << "Device supports cl_khr_fp16 extension" << std::endl;
		}
		else
		{
			std::cout << "Device does not support cl_khr_fp16 extension" << std::endl;
		}

		context = cl::Context(device);
		queue = cl::CommandQueue(context, device);

		size_t maxWorkGroupSize;
		clGetDeviceInfo(device(), CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);

		std::vector<cl_half> r(width * height);
		std::vector<cl_half> g(width * height);
		std::vector<cl_half> b(width * height);
		for (int i = 0; i < height; ++i)
		{
			for (int j = 0; j < width; j++)
			{
				r[i * width + j] = img[(i * width + j) * channels];
				g[i * width + j] = img[(i * width + j) * channels + 1];
				b[i * width + j] = img[(i * width + j) * channels + 2];
			}
		}

		cl::Buffer rBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_half) * r.size(), r.data());
		cl::Buffer gBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_half) * g.size(), g.data());
		cl::Buffer bBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_half) * b.size(), b.data());
		cl::Buffer imageBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_half) * width * height * channels, img.data());
		cl::Buffer darkChannelBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_half) * width * height);
		cl::Buffer atmosphereBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_half) * 3);
		cl::Buffer transEstBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_half) * width * height);
		cl::Buffer radianceBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_half) * width * height * 3);

		cl::Program::Sources sources;
		std::ifstream file("../dehazeApprox.cl");
		std::string source(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
		sources.push_back({source.c_str(), source.length() + 1});

		cl::Program program(context, sources);
		try
		{
			program.build("-cl-std=CL1.2");
		}
		catch (cl::Error &err)
		{
			if (err.err() == CL_BUILD_PROGRAM_FAILURE)
			{
				std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
				std::cerr << "Build log:" << std::endl
						  << buildLog << std::endl;
			}
			throw err;
		}

		cl::Kernel get_dark_channel(program, "get_dark_channel");
		cl::Kernel get_atmosphere(program, "get_atmosphere");
		cl::Kernel get_transmission_estimate(program, "get_transmission_estimate");
		cl::Kernel get_radiance(program, "get_radiance");

		auto start = std::chrono::high_resolution_clock::now();

		queue.enqueueWriteBuffer(imageBuffer, CL_TRUE, 0, sizeof(cl_half) * width * height * channels, img.data());
		std::cout << "Wrote to image buffer" << std::endl;

		get_dark_channel.setArg(0, 0);
		get_dark_channel.setArg(1, rBuffer);
		get_dark_channel.setArg(2, gBuffer);
		get_dark_channel.setArg(3, bBuffer);
		get_dark_channel.setArg(4, height);
		get_dark_channel.setArg(5, width);
		get_dark_channel.setArg(6, 8);
		get_dark_channel.setArg(7, darkChannelBuffer);

		queue.enqueueNDRangeKernel(get_dark_channel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NullRange);

		queue.enqueueReadBuffer(darkChannelBuffer, CL_TRUE, 0, sizeof(cl_half) * darkChannelImg.size(), darkChannelImg.data());

		std::cout << "Dark Channel Image (First 100 elements):" << std::endl;
		for (int i = 0; i < 100; ++i)
		{
			std::cout << darkChannelImg[i] << " ";
		}
		std::cout << std::endl;

		queue.finish();

		get_atmosphere.setArg(0, imageBuffer);
		get_atmosphere.setArg(1, atmosphereBuffer);
		get_atmosphere.setArg(2, width * height);

		queue.enqueueNDRangeKernel(get_atmosphere, cl::NullRange, cl::NDRange(1), cl::NullRange);

		queue.enqueueReadBuffer(atmosphereBuffer, CL_TRUE, 0, sizeof(cl_half) * 3, atmosphere);

		std::cout << "Atmosphere: " << atmosphere[0] << ", " << atmosphere[1] << ", " << atmosphere[2] << std::endl;

		queue.finish();

		get_transmission_estimate.setArg(0, imageBuffer);
		get_transmission_estimate.setArg(1, atmosphereBuffer);
		get_transmission_estimate.setArg(2, transEstBuffer);
		get_transmission_estimate.setArg(3, cl_half(0.5f));
		get_transmission_estimate.setArg(4, height);
		get_transmission_estimate.setArg(5, width);

		queue.enqueueNDRangeKernel(get_transmission_estimate, cl::NullRange, cl::NDRange(width * height), cl::NullRange);

		queue.enqueueReadBuffer(transEstBuffer, CL_TRUE, 0, sizeof(cl_half) * transmissionImg.size(), transmissionImg.data());

		std::cout << "Transmission Estimate (First 100 elements):" << std::endl;
		for (int i = 0; i < 100; ++i)
		{
			std::cout << transmissionImg[i] << " ";
		}
		std::cout << std::endl;

		queue.finish();

		get_radiance.setArg(0, imageBuffer);
		get_radiance.setArg(1, transEstBuffer);
		get_radiance.setArg(2, atmosphereBuffer);
		get_radiance.setArg(3, radianceBuffer);
		get_radiance.setArg(4, width);
		get_radiance.setArg(5, height);

		queue.enqueueNDRangeKernel(get_radiance, cl::NullRange, cl::NDRange(width * height), cl::NullRange);

		std::vector<cl_half> result(width * height * 3);
		queue.enqueueReadBuffer(radianceBuffer, CL_TRUE, 0, sizeof(cl_half) * result.size(), result.data());

		std::cout << "Radiance (First 100 elements):" << std::endl;
		for (int i = 0; i < 100; ++i)
		{
			std::cout << result[i] << " ";
		}
		std::cout << std::endl;

		queue.finish();

		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
		std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;

		// Convert the radiance values from cl_half to float
		std::vector<float> resultFloat(result.size());
		for (int i = 0; i < result.size(); i++)
		{
			resultFloat[i] = result[i];
		}

		// Find the minimum and maximum values in the radiance buffer
		float minVal = *std::min_element(resultFloat.begin(), resultFloat.end());
		float maxVal = *std::max_element(resultFloat.begin(), resultFloat.end());

		// Normalize the radiance values to the range [0, 1]
		for (int i = 0; i < resultFloat.size(); i++)
		{
			resultFloat[i] = (resultFloat[i] - minVal) / (maxVal - minVal);
		}

		// Scale the normalized radiance values to the range [0, 255]
		for (int i = 0; i < resultFloat.size(); i++)
		{
			resultFloat[i] *= 255.0f;
		}

		Mat imgcv_out(height, width, CV_32FC3, resultFloat.data());

		// Apply gamma correction
		float gamma = 1.0f;
		cv::pow(imgcv_out, gamma, imgcv_out);

		// Convert the image from floating-point to unsigned 8-bit
		imgcv_out.convertTo(imgcv_out, CV_8UC3, 1.0, 0);

		// Convert color space from RGB to BGR
		cv::cvtColor(imgcv_out, imgcv_out, cv::COLOR_RGB2BGR);
		imwrite("C:/Users/jpeop/dissertation/csc3002_image_dehazing/csc3002_new/approximateresult.png", imgcv_out);
	}
	catch (cl::Error err)
	{
		std::cerr << "Exception: " << err.what() << " (" << err.err() << ")" << std::endl;
		return 1;
	}

	std::cout << "Press ENTER to exit...";
	std::cin.get();

	return 0;
}