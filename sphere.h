//sphere.h
#ifndef _SPHERE_H_
#define _SPHERE_H_

#include "object.h"
#include "ray.h"
#include "vector3D.h"
#include "color.h"
#include "lightsource.h"

class Sphere : public Object
{
private:
	Vector3D position;
	double radius;

public:
	Sphere(const Vector3D& _pos, double _rad, Material* mat):
		Object(mat), position(_pos), radius(_rad)
	{
		isSolid = true;
	}
	
	virtual bool intersect(Ray& r) const;
};
#endif
