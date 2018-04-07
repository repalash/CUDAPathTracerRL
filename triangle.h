//triangle.h
#ifndef _TRIANGLE_H_
#define _TRIANGLE_H_

#include "object.h"
#include "ray.h"
#include "vector3D.h"
#include "color.h"

class Triangle : public Object
{
private:
	Vector3D a, b, c;
	Vector3D aN, bN, cN; //Normals at the vertices

public:
	Triangle(const Vector3D& _a, const Vector3D& _b, const Vector3D& _c, Material* mat):
			Object(mat), a(_a), b(_b), c(_c)
	{
		aN = bN = cN = crossProduct(_b-_a, _c-_a);
		isSolid = true;
	}
	Triangle(const Vector3D& _a, const Vector3D& _b, const Vector3D& _c, const Vector3D& _nA, const Vector3D& _nB, const Vector3D& _nC, Material* mat):
			Object(mat), a(_a), b(_b), c(_c), aN(_nA), bN(_nB), cN(_nC)
	{
		isSolid = true;
	}

	virtual bool intersect(Ray& r) const;
};
#endif
