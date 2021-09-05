#pragma once
#include"Common.h"

struct HitRecord
{
	Point3 p;
	Vec3 normal;
	float t;
	bool bFront;
	bool bPadding;
	short mPadding;

	inline __device__ void SetFaceNormal(const Ray& r, const Vec3& outwardNormal)
	{
		bFront = Dot(r.mDirection, outwardNormal) < 0;
		normal = bFront ? outwardNormal : -outwardNormal;
	}
};

class Hittable
{

};