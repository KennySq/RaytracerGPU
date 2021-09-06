#include"Common.h"
#include"Sphere.cuh"
#include"HittableList.cuh"

#include<Windows.h>
#include<iostream>

#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif
#include<device_functions.h>
#include<device_launch_parameters.h>
#include<cuda_runtime_api.h>
#include<device_types.h>




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