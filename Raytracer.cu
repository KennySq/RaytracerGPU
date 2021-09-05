#include"Raytracer.cuh"

typedef unsigned int uint;
typedef unsigned char uchar;

LPDWORD gPixels;

__device__ Hittable* deviceScene;
std::vector<Hittable*> hostScene;

void cudaCopyPixels(LPDWORD cpuPixels)
{
	cudaMemcpy((void*)(cpuPixels), (void*)(gPixels), 4 * 800 * 600, cudaMemcpyDeviceToHost);
	std::cout << cudaGetErrorString(cudaGetLastError()) << '\n';
}

__device__ void initCudaRandom(curandState* state)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int seed = id;

	curand_init(seed, 0, 0, &state[id]);
}

__device__ float HitSphere(const Point3& center, float radius, const Ray& r)
{
	Vec3 oc = r.mOrigin - center;

	auto a = LengthSquared(&r.mDirection);

	auto bHalf = Dot(oc, r.mDirection);
	auto c = LengthSquared(&oc) - radius * radius;

	auto discriminant = bHalf * bHalf - a * c;

	if (discriminant < 0)
	{
		return -1.0;
	}
	else
	{
		return (-bHalf - sqrt(discriminant)) / a;
	}
}

template<typename _Ty>
__host__ void GetMappedPointer(_Ty** ptr, unsigned int count)
{
	cudaSetDeviceFlags(cudaDeviceMapHost);

	_Ty* hostMemory;

	cudaHostAlloc((void**)&hostMemory, sizeof(_Ty) * count, cudaHostAllocMapped);

	for (unsigned int i = 0; i < count; i++)
	{
		hostMemory[i] = (_Ty)0;
	}

	_Ty* deviceMemory;
	cudaHostGetDevicePointer((void**)&deviceMemory, (void*)hostMemory, 0);
	


}

void cudaCopyScene(Hittable* deviceScene, std::vector<Hittable*>& host)
{



	for (unsigned int i = 0; i < host.size(); i++)
	{
		void* dst = (void*)(&deviceScene[i]);
		void* src = (void*)(host.data()[i]);


		cudaError_t error = cudaMemcpy(dst, src, sizeof(Hittable*), cudaMemcpyHostToDevice);
#ifdef _DEBUG
		std::cout << cudaGetErrorString(error) << std::endl;
#endif
	}
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

	auto index = __umul24(__umul24(blockIdx.x, blockDim.x), blockDim.y) + __umul24(threadIdx.y, blockDim.x) + threadIdx.x;

	pixels[index] = writeColor;
	__syncthreads();

	return;
}

__device__ void RayColor(LPDWORD pixels, const Ray& r, Hittable* pRawDeviceScene, unsigned int count, unsigned int width, unsigned int height)
{
	HitRecord rec;
	Color pOutColor;
	for (unsigned int i = 0; i < count; i++)
	{
//#ifdef _DEBUG
//
//		printf("%d\n", &pRawDeviceScene[i]);
//#endif
		auto sphere = (Sphere*)(&pRawDeviceScene[i]);

		if(sphere->Hit(r, 0, INF, rec))
		{
			//pOutColor = 0.5 * (rec.normal + Color(1, 1, 1));
			//setColor(pixels, width, height, pOutColor);

			continue;
		}

		Vec3 unitDirection = UnitVector(r.mDirection);

		auto t = 0.5 * (unitDirection.e[1] + 1.0);

		pOutColor = (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);
		setColor(pixels, width, height, pOutColor);
	}

	return;

}


__global__ void CudaRender(LPDWORD pixels, unsigned int width, unsigned int height, Hittable* deviceScene, unsigned int count)
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

	auto u = float(threadIdx.x) / (width - 1);
	auto v = float(blockIdx.x) / (height - 1);

	Ray r(origin, lowerLeft + u * horizontal + v * vertical - origin);

	RayColor(pixels, r, deviceScene, count, width, height);

}

__global__ void ClearGradiant(LPDWORD pixels, unsigned int width, unsigned int height, Color color)
{

	int writeColor = 0;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	auto r = __fdiv_rn(threadIdx.x, (width - 1));
	auto g = __fdiv_ru(blockIdx.x, (height - 1));
	auto b = color.e[2];

	int ir = static_cast<int>(__fmul_rd(255.999, r));
	int ig = static_cast<int>(__fmul_rd(255.999, g));
	int ib = static_cast<int>(__fmul_rd(255.999, b));

	writeColor |= (ir << 16);
	writeColor |= (ig << 8);
	writeColor |= ib;

	auto index = __umul24(__umul24(blockIdx.x, blockDim.x), blockDim.y) + __umul24(threadIdx.y, blockDim.x) + threadIdx.x;

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

	error = cudaMalloc((void**)(&deviceScene), sizeof(Hittable*) * 2);
	std::cout << cudaGetErrorString(error) << '\n';

	error = cudaMemset((void**)(deviceScene), 0, sizeof(Hittable*) * 2);
	std::cout << cudaGetErrorString(error) << '\n';

	hostScene.push_back(new Sphere(Point3(0, 0, -1), 0.5));
	hostScene.push_back(new Sphere(Point3(0, -100.5, -1), 100));

	cudaCopyScene(deviceScene, hostScene);

	//ClearGradiant << <600, 800>> > (gPixels, mWidth, mHeight, Color(1, 1, 0.25));

}

void Raytracer::Run()
{
	//ClearGradiant << <600, 800>> > (gPixels, mWidth, mHeight, Color(1, 1, 0.25));
	auto rawScene = deviceScene;

	CudaRender << <600, 800 >> > (gPixels, mWidth, mHeight, rawScene, hostScene.size());
	cudaDeviceSynchronize();
	std::cout << cudaGetErrorString(cudaGetLastError()) << '\n';
	//cudaThreadSynchronize();

	cudaCopyPixels(mPixels);
}

void Raytracer::Release()
{
	DeleteDC(mMemoryDC);
	DeleteObject(mBitmap);

}

