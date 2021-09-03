#pragma once

#include<cmath>
#include<random>
#include<cuda_runtime.h>
#include<device_functions.h>
#include<cuda/std/cmath>
#include<curand.h>
#include<curand_kernel.h>
#include<curand_uniform.h>


//__device__ double Rand()
//{
//	double randValue;
//	curandState* state;
//	
//	randValue = curand_uniform(state);
//
//	return randValue;
//}
//
//__device__ double Rand(double min, double max)
//{
//	return min + (max - min) * Rand();
//}

class Vec3
{
public:
	__host__ __device__ Vec3() : e{ 0,0,0 } {}
	__host__ __device__ Vec3(double e0, double e1, double e2) : e{ e0,e1,e2 } {}
//
//	double x() const { return e[0]; }
//	double y() const { return e[1]; }
//	double z() const { return e[2]; }
//
	__device__	Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
//	double operator[](int i) const { return e[i]; }
//	double& operator[](int i) { return e[i]; }
//
//	Vec3& operator+=(const Vec3& v)
//	{
//		e[0] += v.e[0];
//		e[1] += v.e[1];
//		e[2] += v.e[2];
//
//		return *this;
//	}
//
	__device__ Vec3& operator*=(const double t)
	{
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;

		return *this;
	}
//
//	Vec3& operator/=(const double t)
//	{
//		return *this *= 1 / t;
//	}
//
//	double Length() const { return sqrt(LengthSqaured()); }
//	double LengthSqaured() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
//
//	inline static Vec3 Radnom()
//	{
//		return Vec3(Rand(), Rand(), Rand());
//	}
//
//	inline static Vec3 Random(double min, double max)
//	{
//		return Vec3(Rand(min, max), Rand(min, max), Rand(min, max));
//	}
public:
	double e[3];
};

using Point3 = Vec3;
using Color = Vec3;

//__device__ Vec3 Radnom()
//{
//	return Vec3(Rand(), Rand(), Rand());
//}
//
//__device__ Vec3 Random(double min, double max)
//{
//	return Vec3(Rand(min, max), Rand(min, max), Rand(min, max));
//}


//__device__  Vec3 RandomUnitSphere()
//{
//	while (true)
//	{
//		auto p = Random(-1, 1);
//		if (LengthSqaured(&p) >= 1) continue;
//		return p;
//	}
//}


inline __device__ 	double LengthSquared(const Vec3* vec)
{
	return vec->e[0] * vec->e[0] + vec->e[1] * vec->e[1] + vec->e[2] * vec->e[2];
}

inline __device__ double Length(const Vec3* vec)
{
	return sqrt(LengthSquared(vec));
}


inline __device__ Vec3 operator+(const Vec3& u, const Vec3& v)
{
	return Vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline __device__ Vec3 operator-(const Vec3& u, const Vec3& v)
{
	return Vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline __device__ Vec3 operator*(const Vec3& u, const Vec3& v)
{
	return Vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}
inline __device__ Vec3 operator*(double t, const Vec3& v)
{
	return Vec3(v.e[0] * t, v.e[1] * t, v.e[2] * t);
}

inline __device__ Vec3 operator*(const Vec3& v, double t)
{
	return t * v;
}

inline __device__ Vec3 operator/(Vec3 v, double t)
{
	return (1 / t) * v;
}

inline __device__ double Dot(const Vec3& u, const Vec3& v)
{
	return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

inline __device__ Vec3 Cross(const Vec3& u, const Vec3& v)
{
	return Vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1], u.e[2] * v.e[0] - u.e[0] * v.e[2], u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline __device__ Vec3 UnitVector(Vec3 v)
{
	return v / Length(&v);
}