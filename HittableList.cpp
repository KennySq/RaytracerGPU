#include "HittableList.h"

bool HittableList::Hit(const Ray& r, double tMin, double tMax, HitRecord& rec)
{
    HitRecord tempRec;

    bool bHitSomething = false;
    auto closest = tMax;
    

    for (unsigned int i = 0; i < mCount; i++)
    {
        if (mRawArray[i]->Hit(r, tMin, closest, tempRec))
        {
            bHitSomething = true;
            closest = tempRec.t;
            rec = tempRec;
        }
    }

    return bHitSomething;
}
