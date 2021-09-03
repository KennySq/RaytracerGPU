#pragma once
#include"Common.h"

struct HitRecord
{
	Point3 p;
	Vec3 normal;
	double t;
	bool bFront;

	__device__ void SetFaceNormal(const Ray& r, const Vec3& outwardNormal)
	{
		bFront = Dot(r.mDirection, outwardNormal) < 0;
		normal = bFront ? outwardNormal : -outwardNormal;
	}
};

class Hittable
{
public:
	__host__ __device__ virtual bool Hit(const Ray& r, double tMin, double tMax, HitRecord& rec) = 0;
};

