#include"Common.h"

class __align__(64) Camera
{
public:
	__device__ Camera()
	{
		const auto aspectRatio = 4.0 / 3.0;

		const float viewportHeight = 2.0;
		const float viewportWidth = viewportHeight * aspectRatio;

		auto origin = Point3(0, 0, 0);
		auto horizontal = Vec3(aspectRatio * 2.0, 0, 0);
		auto vertical = Vec3(0, 2.0, 0);
		auto lowerLeft = origin - horizontal / 2 - vertical / 2 - Vec3(0, 0, 1.0);

	}

	__device__ Ray GetRay(float u, float v)
	{
		return Ray(mOrigin, mLowerLeft + u * mHorizontal + v * mVertical - mOrigin);
	}

public:
	Point3 mOrigin;
	Point3 mLowerLeft;
	Vec3 mHorizontal;
	Vec3 mVertical;
	int mSampleCount;

};