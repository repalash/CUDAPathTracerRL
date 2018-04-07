//triangle.cpp

#include "triangle.h"

bool Triangle::intersect(Ray &r) const {
	Vector3D e1, e2;
	Vector3D N, T;
	double det, alpha, beta;

	e1 = b - a;
	e2 = c - a;

	N = crossProduct(r.getDirection(), e2);
	det = dotProduct(e1, N);
	T = r.getOrigin() - a;
	alpha = dotProduct(T, N) / det;
	N = crossProduct(T, e1);
	beta = dotProduct(r.getDirection(), N) / det;
	if(fabs(det) < 0.0001 || alpha < 0.f || alpha > 1.f || beta < 0.f || alpha + beta  > 1.f)
		return false;
	double t = dotProduct(e2, N) / det;
	if(t > 0.0001) {
		Vector3D normal = alpha*aN + beta*bN + (1-alpha-beta)*cN;
		return r.setParameter(t, this, normal);
	}
	return false;
}

