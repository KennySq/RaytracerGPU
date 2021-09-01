#include"Common.h"
#include<Windows.h>
#include<cuda_runtime.h>
#include<device_functions.h>
#include<device_launch_parameters.h>

#ifdef __cplusplus
extern "C" {
#endif

	class Raytracer
	{
	public:
		Raytracer(HWND handle, HINSTANCE instance, unsigned int width, unsigned int height);

		void Run();

		HDC GetMemoryDC() const { return mMemoryDC; }

		void setColor(int x, int y);

		void Release();
	private:
		unsigned int mWidth;
		unsigned int mHeight;

		HWND mHandle;
		HBITMAP mBitmap;

		HDC mMemoryDC;
		LPDWORD mPixels;

		HINSTANCE mInst;
	};


#ifdef __cplusplus
}
#endif