#pragma once


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