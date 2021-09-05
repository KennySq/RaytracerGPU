#include "HittableList.cuh"

//__host__ __device__ bool HittableList::Hit(const Ray& r, float tMin, float tMax, HitRecord& rec)
//{
//    HitRecord tempRec;
//
//    bool bHitSomething = false;
//    auto closest = tMax;
//    
//
//    for (unsigned int i = 0; i < mCount; i++)
//    {
//        if (mRawArray[i]->Hit(r, tMin, closest, tempRec))
//        {
//            bHitSomething = true;
//            closest = tempRec.t;
//            rec = tempRec;
//        }
//    }
//
//    return bHitSomething;
//}
