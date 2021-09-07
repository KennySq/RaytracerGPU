#pragma once

#include<random>
#include<cuda_runtime.h>
#include<device_functions.h>
#include<cuda/std/cmath>
#include<curand.h>
#include<curand_kernel.h>
#include<curand_uniform.h>


inline __device__ float Rand(curandState* randState, int tid)
{
	float randValue =  0;

	curand_init(tid, 0, 0, &randState[tid]);
	randValue = curand_uniform(&randState[tid]);

	return randValue;
}

inline __device__ float Rand(float min, float max, curandState* randState, int tid)
{
	return min + (max - min) * Rand(randState, tid);
}



typedef struct __align__(16) Vec3
{
public:
	__host__ __device__ Vec3() : e{ 0,0,0 } {}
	__host__ __device__ Vec3(float e0, float e1, float e2) : e{ e0,e1,e2 } {}

	inline __device__ Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
//	float operator[](int i) const { return e[i]; }
//	float& operator[](int i) { return e[i]; }
//
	inline __device__ Vec3& operator+=(const Vec3& v)
	{
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];

		return *this;
	}
//
	inline __device__ Vec3& operator*=(const float t)
	{
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;

		return *this;
	}
//
//	Vec3& operator/=(const float t)
//	{
//		return *this *= 1 / t;
//	}


//


//	inline static Vec3 Radnom()
//	{
//		return Vec3(Rand(), Rand(), Rand());
//	}
//
//	inline static Vec3 Random(float min, float max)
//	{
//		return Vec3(Rand(min, max), Rand(min, max), Rand(min, max));
//	}
public:
	float e[3];
}Point3, Color;


inline __device__ Vec3 RandVec(float min, float max, curandState* randState, int tid)
{
	return Vec3(Rand(min, max, randState, tid), Rand(min, max, randState, tid), Rand(min, max, randState, tid));
}
inline __device__	float LengthSquared(const Vec3& v) { return v.e[0] * v.e[0] + v.e[1] * v.e[1] + v.e[2] * v.e[2]; }

inline __device__	float Length(const Vec3& v) { return sqrt(LengthSquared(v)); }

inline __device__  Vec3 RandomUnitSphere(curandState* randState, int tid)
{
	while (true)
	{
		auto p = RandVec(-1, 1, randState, tid);
//#ifdef _DEBUG
//		printf("%.2f, %.2f, %.2f", p.e[0], p.e[1], p.e[2]);
//#endif
		
		if (LengthSquared(p) >= 1) continue;
		return p;
	}
}
//__device__ Vec3 Radnom()
//{
//	return Vec3(Rand(), Rand(), Rand());
//}
//
//__device__ Vec3 Random(float min, float max)
//{
//	return Vec3(Rand(min, max), Rand(min, max), Rand(min, max));
//}



inline __device__ Vec3 __vsub3(const Vec3& u, const Vec3& v)
{
	return Vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
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
inline __device__ Vec3 operator*(float t, const Vec3& v)
{
	return Vec3(v.e[0] * t, v.e[1] * t, v.e[2] * t);
}

inline __device__ Vec3 operator*(const Vec3& v, float t)
{
	return t * v;
}

inline __device__ Vec3 operator/(Vec3 v, float t)
{
	return (1 / t) * v;
}

inline __device__ float Dot(const Vec3& u, const Vec3& v)
{
	return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

inline __device__ Vec3 Cross(const Vec3& u, const Vec3& v)
{
	return Vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1], u.e[2] * v.e[0] - u.e[0] * v.e[2], u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline __device__ Vec3 UnitVector(const Vec3& v)
{
	return v / Length(v);
}