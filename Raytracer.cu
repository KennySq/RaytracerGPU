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

void ClearGradiantCPU(LPDWORD pixels, unsigned int width, unsigned int height, Color color)
{
	for (int y = height - 1; y >= 0; y--)
	{
		for (int x = 0; x < width; x++)
		{
			int color = 0;
			auto r = double(x) / (width - 1);
			auto g = double(y) / (height - 1);
			auto b = 0.25;

			int ir = static_cast<int>(255.999 * r);
			int ig = static_cast<int>(255.999 * g);
			int ib = static_cast<int>(255.999 * b);

			color |= (ir << 16);
			color |= (ig << 8);
			color |= ib;
			auto index = (y * width) + x;
			pixels[index] = color;
		}
	}
}


__global__ void ClearGradiant(LPDWORD pixels, unsigned int width, unsigned int height, Color color)
{

	int writeColor = 0;



	int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	auto r = double(x) / (width - 1);
	auto g = double(y) / (height - 1);
	auto b = color.e[2];

	int ir = static_cast<int>(255.999 * r);
	int ig = static_cast<int>(255.999 * g);
	int ib = static_cast<int>(255.999 * b);

	writeColor |= (ir << 16);
	writeColor |= (ig << 8);
	writeColor |= ib;

	auto index = __umul24(__umul24(blockIdx.x, blockDim.x), blockDim.y) + __umul24(threadIdx.y, blockDim.x) + threadIdx.x;

	pixels[(y * width) + x] = writeColor;
	
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
//	cudaMemcpyToArray(gPixelArray, 0, 0, reinterpret_cast<void*>(mPixels), sizeof(unsigned short) * 800 * 600, cudaMemcpyHostToDevice);

	ClearGradiant << <600, 800 >> > (gPixels, mWidth, mHeight, Color(1, 1, 0.25));

	cudaCopyPixels(mPixels);
}

void Raytracer::Release()
{
	DeleteDC(mMemoryDC);
	DeleteObject(mBitmap);

}

