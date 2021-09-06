#include"Raytracer.cuh"

typedef unsigned int uint;
typedef unsigned char uchar;

LPDWORD gPixels;
__device__ Sphere* deviceScene;


std::vector<Sphere> hostScene;



__global__ void cudaCopyPixels(LPDWORD cpuPixels, LPDWORD gpuPixels, unsigned int size)
{

	if (threadIdx.x == 0)
	{
		for (unsigned int i = 0; i < size; i++)
		{
			cpuPixels[i] = gpuPixels[i];
		}
	}
	
	__syncthreads();
}

__global__ void cudaInitDeviceMemory()
{
	printf("cudaInitDeviceMemory\n");

	printf("\t - Malloc device scene memory.\n");
	deviceScene = (Sphere*)(malloc(sizeof(Sphere) * 2));

	printf("\t\t Malloc result : %p\n", &deviceScene[0]);
	printf("\t\t Malloc result : %p\n", &deviceScene[1]);

	printf("\t\t %d thread acquire %p \n", threadIdx.x, &deviceScene[0]);
	printf("\t\t %d thread acquire %p \n", threadIdx.x, &deviceScene[1]);
}

__global__ void cudaCopyScene(Sphere* hostScene, unsigned int count)
{
	printf("copy scene (gpu)\n");

	printf("\tdevice object - %p\n", &deviceScene[0]);
	printf("\thost object - %p\n", hostScene[0]);
	if (threadIdx.x == 0)
	{
		for (unsigned int i = 0; i < count; i++)
		{
			deviceScene[i] = hostScene[i];

			printf("%p\n", &deviceScene[i]);
		}
	}
	__syncthreads();
}
__device__ void setColor(LPDWORD pixels, unsigned int width, unsigned int height, Color color)
{
	int writeColor = 0;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	auto r = color.e[0];
	auto g = color.e[1];
	auto b = color.e[2];

	int ir = static_cast<int>(__fmul_rd(255.999, r));
	int ig = static_cast<int>(__fmul_rd(255.999, g));
	int ib = static_cast<int>(__fmul_rd(255.999, b));

	writeColor |= (ir << 16);
	writeColor |= (ig << 8);
	writeColor |= ib;

	auto index = y * width + x;
	pixels[index] = writeColor;
	__syncthreads();

	return;
}

__device__ void RayColor(LPDWORD pixels, const Ray& r, unsigned int count, unsigned int width, unsigned int height)
{
	HitRecord rec;
	Color pOutColor;
//#ifdef _DEBUG
//
//		printf("%d\n", &pRawDeviceScene[i]);
//#endif
	Sphere sphere = deviceScene[gridDim.z];

	//if(sphere.Hit(r, 0, INF, rec))
	//{
	//	pOutColor = 0.5 * (rec.normal + Color(1, 1, 1));
	//	setColor(pixels, width, height, pOutColor);
	//	return;
	//}

	Vec3 unitDirection = UnitVector(r.mDirection);

	auto t = 0.5 * (unitDirection.e[1] + 1.0);
	pOutColor = (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);
	setColor(pixels, width, height, pOutColor);

	return;
}


__global__ void CudaRender(LPDWORD pixels, unsigned int width, unsigned int height, unsigned int count)
{

	const auto aspectRatio = 4.0 / 3.0;
	const int imageWidth = width;
	const int imageHeight = height;

	auto origin = Point3(0, 0, 0);
	auto horizontal = Vec3(aspectRatio * 2.0, 0, 0);
	auto vertical = Vec3(0, 2.0, 0);
	auto lowerLeft = origin - horizontal / 2 - vertical / 2 - Vec3(0, 0, 1.0);

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	auto u = float(x) / (width - 1);
	auto v = float(y) / (height - 1);

	Ray r(origin, lowerLeft + u * horizontal + v * vertical - origin);

	RayColor(pixels, r, count, width, height);

	__syncthreads();
}

__global__ void ClearGradiant(LPDWORD pixels, unsigned int width, unsigned int height, Color color)
{

	int writeColor = 0;

	//int x = blockIdx.x * blockDim.x + threadIdx.x;
	//int y = blockIdx.y * blockDim.y + threadIdx.y;

	//auto r = __fdiv_rn(threadIdx.x, (width - 1));
	//auto g = __fdiv_ru(blockIdx.x, (height - 1));
 
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	auto r = __fdiv_rn(x, (width - 1));
	auto g = __fdiv_ru(y, (height - 1));
	auto b = color.e[2];

	int ir = static_cast<int>(__fmul_rd(255.999, r));
	int ig = static_cast<int>(__fmul_rd(255.999, g));
	int ib = static_cast<int>(__fmul_rd(255.999, b));

	writeColor |= (ir << 16);
	writeColor |= (ig << 8);
	writeColor |= ib;

	auto index = y * width + x;
	pixels[index] = writeColor;
	__syncthreads();

	return;
}



Raytracer::Raytracer(HWND handle, HINSTANCE instance, unsigned int width, unsigned int height)
	: mHandle(handle), mInst(instance), mWidth(width), mHeight(height)
{
	cudaDeviceProp prop;
	cudaError_t error = cudaGetDeviceProperties(&prop, 0);
	std::cout << cudaGetErrorString(error) << std::endl;
	BITMAPINFO bitInfo{};

	bitInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bitInfo.bmiHeader.biWidth = width;
	bitInfo.bmiHeader.biHeight = height;
	bitInfo.bmiHeader.biBitCount = 32;
	bitInfo.bmiHeader.biPlanes = 1;
	bitInfo.bmiHeader.biCompression = BI_RGB;

	HDC dc = GetDC(mHandle);

	mBitmap = CreateDIBSection(dc, &bitInfo, DIB_RGB_COLORS, (void**)(&mPixels), nullptr, 0);

	mMemoryDC = CreateCompatibleDC(dc);

	SelectObject(mMemoryDC, mBitmap);
	ReleaseDC(mHandle, dc);

	error = cudaMalloc((void**)(&gPixels), 4 * 800 * 600);
	std::cout << cudaGetErrorString(error) << '\n';

	//error = cudaMalloc((void**)(&deviceScene), sizeof(Hittable*) * 2);
	//std::cout << cudaGetErrorString(error) << '\n';

	//error = cudaMemset((void**)(deviceScene), 0, sizeof(Hittable*) * 2);
	//std::cout << cudaGetErrorString(error) << '\n';

	hostScene.push_back(Sphere(Point3(0, 0, -1), 0.5));
	hostScene.push_back(Sphere(Point3(0, -100.5, -1), 100));

	//cudaInitDeviceMemory << <1, 1>> > ();

	cudaDeviceSynchronize();
	//cudaCopyScene<<<1,1>>>(hostScene.data(), hostScene.size());

	printf("Start malloc device memory.\n");
	error = cudaMalloc((void**)&deviceScene, sizeof(Sphere) * 2);
	if (error != cudaError::cudaSuccess)
	{
		printf("critical error occured, result must be cudaSuccess.\n");
		printf("%s\n", cudaGetErrorString(error));
		terminate();
		//throw std::runtime_error("");
	}
	printf("\t - Success.\n");

	printf("Start copying host memory to device.\n");
	error = cudaMemcpy(deviceScene, hostScene.data(), sizeof(Sphere) * 2, cudaMemcpyDeviceToHost);

	if (error != cudaError::cudaSuccess)
	{
		printf("critical error occured, result must be cudaSuccess.\n");
		printf("%s\n", cudaGetErrorString(error));
		terminate();
		//throw std::runtime_error("");
	}
	printf("\t - Success.\n");

	cudaDeviceSynchronize();

	//ClearGradiant << <600, 800>> > (gPixels, mWidth, mHeight, Color(1, 1, 0.25));

}

void Raytracer::Run()
{
	dim3 blocks = dim3(16, 12, 1);
	dim3 grids = dim3(800 / blocks.x, 600 / blocks.y, hostScene.size());
	cudaError error;
	//ClearGradiant << <grids, blocks>> > (gPixels, mWidth, mHeight, Color(1, 1, 0.25));

	CudaRender << <grids, blocks>> > (gPixels, mWidth, mHeight, hostScene.size());
	error = cudaGetLastError();
	
	if (error != cudaError::cudaSuccess)
	{
		std::cerr << "critical error occured, result must be cudaSuccess.\n";
		std::cerr << cudaGetErrorString(error) << std::endl;

		throw std::runtime_error("");
	}
	//cudaCopyPixels<<<1,1>>>(mPixels,gPixels, 800 * 600);

	error = cudaMemcpy((void*)mPixels, (void*)gPixels, sizeof(DWORD) * 800 * 600, cudaMemcpyDeviceToHost);
	if (error != cudaError::cudaSuccess)
	{
		std::cerr << "critical error occured, result must be cudaSuccess.\n";
		std::cerr << cudaGetErrorString(error) << std::endl;

		throw std::runtime_error("");
	}

}

void Raytracer::Release()
{
	DeleteDC(mMemoryDC);
	DeleteObject(mBitmap);

}

