#include "ray.h"

bool Ray::setParameter(const float par, const Object *obj, const Vector3D norm)
{
	if(par < t && par > SMALLEST_DIST)
	{
		hit = true;
		t = par;
		object = obj;
		normal = norm;
		normal.normalize();
		//if(dotProduct(direction, normal) > 0) normal = -normal;
		return true;
	}
	return false;
}
