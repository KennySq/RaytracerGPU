#include"Raytracer.cuh"



typedef unsigned int uint;
typedef unsigned char uchar;

LPDWORD gPixels;
//cudaArray* gPixelArray;
//texture<unsigned short, cudaTextureType2D, cudaReadModeNormalizedFloat> gTexture;

void cudaCopyPixels(LPDWORD cpuPixels)
{
	
	cudaMemcpy(reinterpret_cast<void*>(cpuPixels), reinterpret_cast<void*>(gPixels), 4*800 * 600, cudaMemcpyDeviceToHost);
	std::cout << cudaGetErrorString(cudaGetLastError()) << '\n';
	/*
	auto error = cudaMemcpyFromArray(reinterpret_cast<void*>(cpuPixels), gPixelArray, 0,0,3 * 800 * 600, cudaMemcpyDeviceToHost);
	std::cout << cudaGetErrorString(error) << std::endl;*/
}



//void ClearGradiantCPU(LPDWORD pixels, unsigned int width, unsigned int height, Color color)
//{
//	for (int y = height - 1; y >= 0; y--)
//	{
//		for (int x = 0; x < width; x++)
//		{
//			int color = 0;
//			auto r = double(x) / (width - 1);
//			auto g = double(y) / (height - 1);
//			auto b = 0.25;
//
//			int ir = static_cast<int>(255.999 * r);
//			int ig = static_cast<int>(255.999 * g);
//			int ib = static_cast<int>(255.999 * b);
//
//			color |= (ir << 16);
//			color |= (ig << 8);
//			color |= ib;
//			auto index = (y * width) + x;
//			pixels[index] = color;
//		}
//	}
//}
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
__device__ void RayColor(Color& pOutColor, const Ray& r)
{
	auto t = HitSphere(Point3(0, 0, -1), 0.5, r);
	
	if (t > 0.0)
	{
		Vec3 n = UnitVector(r.At(t) - Vec3(0, 0, -1));
		pOutColor = 0.5 * Color(n.e[0] + 1, n.e[1] + 1, n.e[2] + 1);
		return;
	}

	Vec3 unitDirection = UnitVector(r.mDirection);

	t = 0.5 * (unitDirection.e[1] + 1.0);

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

	auto viewportHeight = 2.0;
	auto viewportWidth = aspectRatio * viewportHeight;
	auto focalLength = 1.0;

	auto origin = Point3(0, 0, 0);
	auto horizontal = Vec3(viewportWidth, 0, 0);
	auto vertical = Vec3(0, viewportHeight, 0);
	auto lowerLeft = origin - horizontal / 2 - vertical / 2 - Vec3(0, 0, focalLength);

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

	auto r =  __fdiv_rn(threadIdx.x, (width - 1));
	auto g =  __fdiv_ru(blockIdx.x, (height - 1));
	auto b =  color.e[2];

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

	cudaMalloc(&gPixels, 4*800 * 600);


}

void Raytracer::Run()
{
	//ClearGradiantCPU(mPixels, mWidth, mHeight, Color(1,0,0));

	//ClearGradiant << <600, 800>> > (gPixels, mWidth, mHeight, Color(1, 1, 0.25));
	CudaRender << <600, 800 >> > (gPixels, mWidth, mHeight);


	cudaCopyPixels(mPixels);
}

void Raytracer::Release()
{
	DeleteDC(mMemoryDC);
	DeleteObject(mBitmap);

}

