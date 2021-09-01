#include"Raytracer.cuh"

__global__ void ClearGradiant(LPDWORD pixels, unsigned int width, unsigned int height, Color color)
{
	int writeColor = 0;

	auto r = color.e[0];
	auto g = color.e[1];
	auto b = color.e[2];

	//auto scale = 1.0 / SampleCount;

	//r *= scale;
	//g *= scale;
	//b *= scale;

	int ir = static_cast<int>(255.999 * r);
	int ig = static_cast<int>(255.999 * g);
	int ib = static_cast<int>(255.999 * b);

	writeColor |= (ir << 16);
	writeColor |= (ig << 8);
	writeColor |= ib;

	int x = threadIdx.x;
	//int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = threadIdx.y;
	//int y = blockIdx.y * blockDim.y + threadIdx.y;

	pixels[(y * width) + x] = writeColor;

	return;
}

Raytracer::Raytracer(HWND handle, HINSTANCE instance, unsigned int width, unsigned int height)
	: mHandle(handle), mInst(instance), mWidth(width), mHeight(height)
{
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
}

void Raytracer::Run()
{
	ClearGradiant << <16, 16 >> > (mPixels, mWidth, mHeight, Color(1, 0, 0));
}

void Raytracer::Release()
{
	DeleteDC(mMemoryDC);
	DeleteObject(mBitmap);

}


