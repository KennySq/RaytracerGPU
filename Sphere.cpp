#include "Sphere.h"

bool Sphere::Hit(const Ray& r, double tMin, double tMax, HitRecord& rec)
{
	Vec3 oc = r.mOrigin - mCenter;

	auto a = LengthSquared(&r.mDirection);

	auto bHalf = Dot(oc, r.mDirection);
	auto c = LengthSquared(&oc) - mRadius * mRadius;

	auto discriminant = bHalf * bHalf - a * c;

	if (discriminant < 0)
	{
		return false;
	}

	auto sqrtd = sqrt(discriminant);

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
