#pragma once
#include"Hittable.cuh"

// 16 byte
class Sphere
{
public:
	Sphere() {}
	Sphere(Point3 cen, float r) : mCenter(cen), mRadius(r) {}

	inline __device__ bool Hit(const Ray& r, float tMin, float tMax, HitRecord& rec)
	{
		Vec3 oc = r.mOrigin - mCenter;
		
		auto a = LengthSquared(&r.mDirection);
		
		auto bHalf = Dot(oc, r.mDirection);
		
		auto discriminant = bHalf * bHalf - a * (LengthSquared(&oc) - mRadius * mRadius);
		
		if (discriminant < 0)
		{
			return false;
		}
	

		auto sqrtd = cuda::std::sqrtf(discriminant);
		
		auto root = (-bHalf - sqrtd) / a;
		if (root < tMin || tMax < root)
		{
			root = (-bHalf + sqrtd) / a;
			if (root < tMin || tMax < root)
			{
				return false;
			}
		}
		
		rec.t = root;
		rec.p = r.At(rec.t);
		
		Vec3 outwardNormal = (rec.p - mCenter) / mRadius;
		rec.SetFaceNormal(r, outwardNormal);
		
		return true;
	}

	Point3 mCenter;
	float mRadius;
};