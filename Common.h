#pragma once
#include<limits>
#include<cmath>
#include<memory>

#include<cuda.h>
#include<cuda_runtime.h>
#include<thrust/device_vector.h>
using std::shared_ptr;
using std::make_shared;

__constant__ const float INF = std::numeric_limits<float>::infinity();
__constant__ const float PI = 3.1415926535897932385;

inline __device__ float Deg2Rad(float degrees)
{
	return degrees * PI / 180.0;
}


inline float Clamp(float x, float min, float max)
{
	if (x < min)
	{
		return min;
	}
	if (x > max)
	{
		return max;
	}

	return x;

}

#include"Vec3.h"
#include"Ray.cuh"