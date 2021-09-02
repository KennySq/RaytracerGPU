#include"Raytracer.cuh"
extern __shared__ DWORD gPixels[];
__shared__ LPDWORD gpuPixels;
__shared__ LPDWORD gpuPixels2;
texture<int, 2> texArray;
__global__ void cudaMakePixels()
{

	if (threadIdx.x == 0)
	{
		gpuPixels = (LPDWORD)gPixels;
		gpuPixels2 = (LPDWORD)&gpuPixels[479999];
	}

	__syncthreads();
}

__global__ void cudaMakeDIB(HWND handle, HBITMAP bitmap, HDC dc, HDC memoryDC, unsigned int width, unsigned int height)
{
	BITMAPINFO bitInfo{};

	bitInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bitInfo.bmiHeader.biWidth = width;
	bitInfo.bmiHeader.biHeight = height;
	bitInfo.bmiHeader.biBitCount = 32;
	bitInfo.bmiHeader.biPlanes = 1;
	bitInfo.bmiHeader.biCompression = BI_RGB;

	//bitmap = CreateDIBSection(dc, &bitInfo, DIB_RGB_COLORS, reinterpret_cast<void**>(&gPixels), nullptr, 0);

	//memoryDC = CreateCompatibleDC(dc);

	//SelectObject(memoryDC, bitmap);
	//ReleaseDC(handle, dc);

	return;
}

__host__ void cudaCopyBitmap(LPDWORD cpuPixels)
{
	cudaMemcpy(cpuPixels, gpuPixels, sizeof(DWORD) * 4800000, cudaMemcpyKind::cudaMemcpyDeviceToHost);
}
void ClearGradiantCPU(unsigned int width, unsigned int height, Color color)
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

		//	gPixels[(y * width) + x] = color;
		}
	}
}


__global__ void ClearGradiant(LPDWORD pixels, unsigned int width, unsigned int height, Color color)
{

	int writeColor = 0;

	auto r = color.e[0];
	auto g = color.e[1];
	auto b = color.e[2];

	int ir = static_cast<int>(255.999 * r);
	int ig = static_cast<int>(255.999 * g);
	int ib = static_cast<int>(255.999 * b);

	writeColor |= (ir << 16);
	writeColor |= (ig << 8);
	writeColor |= ib;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	//gPixels[(y * width) + x] = writeColor;
	auto index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	printf("%d\n", index);
	// blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x

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

//	cudaMakeDIB<<<1,1>>>(handle, mBitmap, dc, mMemoryDC, width, height);

	mBitmap = CreateDIBSection(dc, &bitInfo, DIB_RGB_COLORS, reinterpret_cast<void**>(&mPixels), nullptr, 0);

	mMemoryDC = CreateCompatibleDC(dc);

	SelectObject(mMemoryDC, mBitmap);
	ReleaseDC(mHandle, dc);

	cudaMakePixels<<<1,1, 49152>>>();
}

void Raytracer::Run()
{


//	ClearGradiantCPU(mWidth, mHeight, Color(1,0,0));
	auto error = cudaGetLastError();
	std::cout << cudaGetErrorString(error) << std::endl;
	
	ClearGradiant << <6, 8>> > (gpuPixels, mWidth, mHeight, Color(1, 1, 0));
	//cudaThreadSynchronize();
	cudaCopyBitmap(mPixels);
}

void Raytracer::Release()
{
	DeleteDC(mMemoryDC);
	DeleteObject(mBitmap);

}

