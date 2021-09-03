#pragma once
#include<limits>
#include<cmath>
#include<memory>

#include<cuda_runtime.h>
#include<thrust/device_vector.h>
using std::shared_ptr;
using std::make_shared;

__constant__ const double INF = std::numeric_limits<double>::infinity();
__constant__ const double PI = 3.1415926535897932385;

inline __device__ double Deg2Rad(double degrees)
{
	return degrees * PI / 180.0;
}


inline double Clamp(double x, double min, double max)
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
#include"Ray.h"