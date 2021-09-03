#pragma once
#include"Hittable.h"

using namespace std;

class HittableList : public Hittable
{
public:
	HittableList() : mCount(0), mCapacity(4) {
		mRawArray = reinterpret_cast<Hittable**>(malloc(mCapacity * sizeof(Hittable*)));
	}
	HittableList(Hittable* object) : mCount(0), mCapacity(4), mRawArray(new Hittable* [mCapacity]) { Add(object); }

	void Clear()
	{
		mCount = 0;
		delete[] mRawArray;
		mRawArray = nullptr;
	}

	void Add(Hittable* object)
	{ 
		if (mCount >= mCapacity)
		{
			unsigned int prevCap = mCapacity;
			mCapacity *= 2;
			Hittable** temp = new Hittable*[mCapacity];

			for (unsigned int i = 0; i < prevCap; i++)
			{
				temp[i] = mRawArray[i];
			}

			delete[] mRawArray;

			mRawArray = temp;
		}

		mRawArray[mCount] = object;
		mCount++;

	}
	
	__device__ virtual bool Hit(const Ray& r, double tMin, double tMax, HitRecord& rec) override;
public:
	unsigned int mCount;
	unsigned int mCapacity;
	Hittable** mRawArray;
};

