#include"Raytracer.cuh"



typedef unsigned int uint;
typedef unsigned char uchar;

LPDWORD gPixels;


void cudaCopyPixels(LPDWORD cpuPixels)
{
	cudaMemcpy(reinterpret_cast<void*>(cpuPixels), reinterpret_cast<void*>(gPixels), 4 * 800 * 600, cudaMemcpyDeviceToHost);
	std::cout << cudaGetErrorString(cudaGetLastError()) << '\n';
}

__device__ void initCudaRandom(curandState* state)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int seed = id;

	curand_init(seed, 0, 0, &state[id]);
}

__device__ double HitSphere(const Point3& center, double radius, const Ray& r)
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

__global__ void AddSphere(Vec3 center, double radius)
{
	if (threadIdx.x == 0)
	{
		Sphere* object;

		cudaMalloc(reinterpret_cast<void**>(&object), sizeof(Sphere));

		unsigned int count = gWorld->mCount;
		unsigned int capacity = gWorld->mCapacity;

		gWorld->Add(object);


	}

	__syncthreads();

}

__device__ void RayColor(Color& pOutColor, const Ray& r)
{
	HitRecord rec;

	if (gWorld->Hit(r, 0, INF, rec))
	{
		pOutColor = 0.5 * (rec.normal + Color(1, 1, 1));
		return;
	}

	Vec3 unitDirection = UnitVector(r.mDirection);

	auto t = 0.5 * (unitDirection.e[1] + 1.0);

	pOutColor = (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);

	return;

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

__global__ void CudaRender(LPDWORD pixels, unsigned int width, unsigned int height)
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

	auto u = double(threadIdx.x) / (width - 1);
	auto v = double(blockIdx.x) / (height - 1);

	Ray r(origin, lowerLeft + u * horizontal + v * vertical - origin);
	Color outColor;

	RayColor(outColor, r);
	setColor(pixels, width, height, outColor);

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
	std::cout << cudaGetErrorString(error) << std::endl;;
	BITMAPINFO bitInfo{};

	bitInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bitInfo.bmiHeader.biWidth = width;
	bitInfo.bmiHeader.biHeight = height;
	bitInfo.bmiHeader.biBitCount = 32;
	bitInfo.bmiHeader.biPlanes = 1;
	bitInfo.bmiHeader.biCompression = BI_RGB;

	HDC dc = GetDC(mHandle);

	mBitmap = CreateDIBSection(dc, &bitInfo, DIB_RGB_COLORS, reinterpret_cast<void**>(&mPixels), nullptr, 0);

	mMemoryDC = CreateCompatibleDC(dc);

	SelectObject(mMemoryDC, mBitmap);
	ReleaseDC(mHandle, dc);

	cudaMalloc(&gPixels, 4 * 800 * 600);
	std::cout << cudaGetErrorString(cudaGetLastError()) << '\n';

	cudaMalloc(reinterpret_cast<void**>(&gWorld), sizeof(Hittable) * 4);

	AddSphere << <1, 1 >> > (Point3(0, 0, -1), 0.5);
	std::cout << cudaGetErrorString(cudaGetLastError()) << '\n';
	AddSphere << <1, 1 >> > (Point3(0, -100.5, -1), 100);
	std::cout << cudaGetErrorString(cudaGetLastError()) << '\n';
	//gWorld->Add(new Sphere(Point3(0, 0, -1), 0.5));
	//gWorld->Add(new Sphere(Point3(0, -100.5, -1), 100));



}

void Raytracer::Run()
{
	//ClearGradiantCPU(mPixels, mWidth, mHeight, Color(1,0,0));

	//ClearGradiant << <600, 800>> > (gPixels, mWidth, mHeight, Color(1, 1, 0.25));
	CudaRender << <600, 800 >> > (gPixels, mWidth, mHeight);
	std::cout << cudaGetErrorString(cudaGetLastError()) << '\n';


	cudaCopyPixels(mPixels);
}

void Raytracer::Release()
{
	DeleteDC(mMemoryDC);
	DeleteObject(mBitmap);

}

