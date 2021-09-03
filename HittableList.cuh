#pragma once
#include"Hittable.h"

using namespace std;

class HittableList : public Hittable
{
public:
	__device__ HittableList() : mCount(0), mCapacity(4) {
		cudaMalloc(reinterpret_cast<void**>(mRawArray), (mCapacity * sizeof(Hittable*)));
		
		cudaChannelFormatDesc formatDesc{};

		
		//cudaMallocArray(&mRawArray, &formatDesc, mCapacity, 1, 0);
	
	}
	//HittableList(Hittable* object) : mCount(0), mCapacity(4), mRawArray(new Hittable* [mCapacity]) { Add(object); }

	//void Clear()
	//{
	//	mCount = 0;
	//	delete[] mRawArray;
	//	mRawArray = nullptr;
	//}

	__device__ void Add(Hittable* object)
	{ 
		//if (mCount >= mCapacity)
		//{
		//	unsigned int prevCap = mCapacity;
		//	mCapacity *= 2;

		//	Hittable** temp;
		//	cudaMalloc((void**)temp, sizeof(Hittable*) * mCapacity);

		//	for (unsigned int i = 0; i < prevCap; i++)
		//	{
		//		temp[i] = mRawArray[i];
		//	}

		//	//cudaFree((void*)mRawArray);

		//	mRawArray = temp;
		//}

		//mRawArray[mCount] = object;
		//mCount++;

	}
	
	__host__ __device__ virtual bool Hit(const Ray& r, double tMin, double tMax, HitRecord& rec) override;
public:
	unsigned int mCount;
	unsigned int mCapacity;
	Hittable** mRawArray;
};

