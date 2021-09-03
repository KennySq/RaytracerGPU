#pragma once
#include"Hittable.h"
class Sphere : public Hittable
{
public:
	Sphere() {}
	Sphere(Point3 cen, double r) : mCenter(cen), mRadius(r) {}

	__host__ __device__ virtual bool Hit(const Ray& r, double tMin, double tMax, HitRecord& rec) override;

public:
	Point3 mCenter;
	double mRadius;
};