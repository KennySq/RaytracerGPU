#include"Raytracer.cuh"

typedef unsigned int uint;
typedef unsigned char uchar;

__device__ LPDWORD gPixels;
__device__ Sphere* deviceScene;
__device__ Camera* deviceCamera;

__device__ curandState* deviceRandState;

const int sampleCount = 1;

std::vector<Sphere> hostScene;
Camera hostCamera;



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
__device__ float getAlpha(LPDWORD pixels, unsigned int width)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int index = y * width + x;

	// alpha mask
	return pixels[index] && -16777216;
}

template<typename _Ty>
void mallocDevice(void** dst, unsigned int count)
{
	cudaError error = cudaMalloc(dst, sizeof(_Ty) * count);
	if (error != cudaError::cudaSuccess)
	{
		printf("\tcritical error occured, result must be cudaSuccess.\n");
		printf("%s\n", cudaGetErrorString(error));
		throw std::runtime_error("");
	}
}

template<typename _Ty>
void copyHostToDevice(_Ty* device, _Ty* host, unsigned int count)
{
	cudaError error = cudaMemcpy(device, host, sizeof(_Ty) * count, cudaMemcpyHostToDevice);
	if (error != cudaError::cudaSuccess)
	{
		printf("\tcritical error occured, result must be cudaSuccess.\n");
		printf("\t%s\n", cudaGetErrorString(error));
		terminate();
		throw std::runtime_error("");
	}
}

__device__ void setColor(LPDWORD pixels, unsigned int width, unsigned int height, Color color, float alpha, int sampleCount)
{
	int writeColor = 0;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	/*auto scale = 1.0 / sampleCount;
	auto r = color.e[0] * scale;
	auto g = color.e[1] * scale;
	auto b = color.e[2] * scale;
	*/

	auto scale = 1.0 / sampleCount;
	auto a = Clamp(alpha, 0, 0.999);
	auto r = Clamp(color.e[0] * scale, 0, 0.999);
	auto g = Clamp(color.e[1] * scale, 0, 0.999);
	auto b = Clamp(color.e[2] * scale, 0, 0.999);

	int ia = static_cast<int>(__fmul_rd(255.999, a));
	int ir = static_cast<int>(__fmul_rd(255.999, r));
	int ig = static_cast<int>(__fmul_rd(255.999, g));
	int ib = static_cast<int>(__fmul_rd(255.999, b));

	// 비트시프트로 채널마다 값 할당
	writeColor |= (ia << 32);
	writeColor |= (ir << 16);
	writeColor |= (ig << 8);
	writeColor |= ib;

	auto index = y * width + x;
	pixels[index] = writeColor;
	__syncthreads();

	return;
}

__global__ void clearPixels(LPDWORD pixels, unsigned int width, unsigned int height, int sampleCount)
{
	const auto aspectRatio = 4.0 / 3.0;
	const int imageWidth = width;
	const int imageHeight = height;

	auto origin = Point3(0, 0, 0);
	auto horizontal = Vec3(aspectRatio * 2.0, 0, 0);
	auto vertical = Vec3(0, 2.0, 0);
	auto lowerLeft = origin - horizontal / 2 - vertical / 2 - Vec3(0, 0, 1.0);

	int x = blockIdx.x * blockDim.x + threadIdx.x * blockIdx.z;
	int y = blockIdx.y * blockDim.y + threadIdx.y * blockIdx.z;

	auto u = float(x) / (width - 1);
	auto v = float(y) / (height - 1);

	Ray r(origin, lowerLeft + u * horizontal + v * vertical - origin);


	Color outColor;
	Vec3 unitDirection = UnitVector(r.mDirection);

	auto t = 0.5 * (unitDirection.e[1] + 1.0);
	outColor = (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);
	setColor(pixels, width, height, outColor, 0, sampleCount);
}

__device__ Color RayColor(LPDWORD pixels, Ray& r, unsigned int count, unsigned int width, unsigned int height, Sphere* deviceScene, int depth, int tid, curandState* randState)
{

	Sphere sphere = deviceScene[blockIdx.z];
	Color outColor{};

	for (unsigned int j = 0; j < blockDim.z; j++)
	{
		Ray curRay = r;
		float atten = 1.0f;

		sphere = deviceScene[j];

		for (unsigned int i = 0; i < depth; i++)
		{
			HitRecord rec{};

			if (sphere.Hit(curRay, 0.001f, INF, rec))
			{
				Point3 target = rec.p + rec.normal + RandomUnitSphere(randState, tid);

				atten *= 0.5;
				//outColor *= atten;

				curRay = Ray(rec.p, target - rec.p);
				//return Color(1, 1, 1);
			}
		}

		Vec3 unitDirection = UnitVector(curRay.mDirection);

		auto t = 0.5 * (unitDirection.e[1] + 1.0);
		outColor = (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);

		return atten * outColor;
		
		

	}
	return Color(0,0,0);
}


__global__ void CudaRender(LPDWORD pixels, unsigned int width, unsigned int height, unsigned int count, Sphere* deviceScene, int sampleCount,curandState* randState)
{
	const auto aspectRatio = 4.0 / 3.0;
	const int imageWidth = width;
	const int imageHeight = height;
	const int depth = 50;

	auto origin = Point3(0, 0, 0);
	auto horizontal = Vec3(aspectRatio * 2.0, 0, 0);
	auto vertical = Vec3(0, 2.0, 0);
	auto lowerLeft = origin - horizontal / 2 - vertical / 2 - Vec3(0, 0, 1.0);

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int tid = y * width + x;

	Color outColor{};

	for (unsigned int i = 0; i < sampleCount; i++)
	{
		auto u = float(x) / (width - 1);
		auto v = float(y) / (height - 1);

		Ray r(origin, lowerLeft + u * horizontal + v * vertical - origin);

		outColor += RayColor(pixels, r, count, width, height, deviceScene, depth, tid, randState);
	//	setColor(pixels, width, height, outColor, 1, 1);
	}

	setColor(pixels, width, height, outColor, 1, sampleCount);

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

__global__ void cudaInitRand(curandState* deviceRandStates, int count, unsigned int width)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int tid = y * width + x;

	curand_init(tid, 0, 0, deviceRandStates);

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


	// CUDA CODE ------------------------------------------------------
	error = cudaMalloc((void**)(&gPixels), 4 * 800 * 600);
	std::cout << cudaGetErrorString(error) << '\n';

	hostScene.push_back(Sphere(Point3(0, 0, -1), 0.5));
	hostScene.push_back(Sphere(Point3(0, -100.5, -1), 100));

	printf("Start malloc device memory.\n");
	mallocDevice<Sphere>((void**)&deviceScene, 2);
	mallocDevice<Camera>((void**)&deviceCamera, 1);

	dim3 blocks = dim3(16, 12, 2);
	dim3 grids = dim3(width / blocks.x, height / blocks.y, 1);

	int threadCount = grids.x * grids.y * blocks.x * blocks.y * blocks.z;
	printf("%d\n", threadCount);
	mallocDevice<curandState>((void**)&deviceRandState, threadCount);
	cudaDeviceSynchronize();

	cudaInitRand<<<grids,blocks>>>(deviceRandState, width, threadCount);

	cudaDeviceSynchronize();
	printf("\t - Success.\n");
	printf("Start copying host memory to device.\n");

	copyHostToDevice<Sphere>(deviceScene, &hostScene[0], 2);
	copyHostToDevice<Camera>(deviceCamera, &hostCamera, 1);

	printf("\t - Success.\n");

	//ClearGradiant << <600, 800>> > (gPixels, mWidth, mHeight, Color(1, 1, 0.25));

}

void Raytracer::Run()
{
	dim3 blocks = dim3(16, 12, hostScene.size());
	dim3 grids = dim3(800 / blocks.x, 600 / blocks.y, 1);
	cudaError error;
	//ClearGradiant << <grids, blocks>> > (gPixels, mWidth, mHeight, Color(1, 1, 0.25));

	//clearPixels << <grids, blocks >> > (gPixels, mWidth, mHeight, sampleCount);

	cudaDeviceSynchronize();

	CudaRender << <grids, blocks>> > (gPixels, mWidth, mHeight, hostScene.size(), deviceScene, sampleCount, deviceRandState);
	error = cudaGetLastError();
	cudaDeviceSynchronize();

	if (error != cudaError::cudaSuccess)
	{
		std::cerr << "\tcritical error occured, result must be cudaSuccess.\n";
		std::cerr << cudaGetErrorName(error) << '\n' << cudaGetErrorString(error) << std::endl;

		throw std::runtime_error("");
	}

	error = cudaMemcpy(mPixels, gPixels, sizeof(DWORD) * 800 * 600, cudaMemcpyDeviceToHost);
	if (error != cudaError::cudaSuccess)
	{
		std::cerr << "\tcritical error occured, result must be cudaSuccess.\n";
		std::cerr << cudaGetErrorString(error) << std::endl;

		throw std::runtime_error("");
	}
	cudaDeviceSynchronize();

}

void Raytracer::Release()
{
	DeleteDC(mMemoryDC);
	DeleteObject(mBitmap);

}

