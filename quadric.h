//quadric.h
#ifndef _QUADRIC_H_
#define _QUADRIC_H_

#include "object.h"
#include "ray.h"
#include "vector3D.h"
#include "color.h"
#include "lightsource.h"

class Quadric : public Object
{
private:
	double A, B, C, D, E, F, G, H, I, J;
	bool out;

public:

	Quadric(double A, double B, double C, double D, double E, double F, double G, double H, double I,
	        double J, bool out, Material* mat) : Object(mat), A(A), B(B), C(C), D(D), E(E), F(F), G(G), H(H), I(I), J(J), out(out) {
		isSolid = true;
		lightSource = nullptr;
	}

	virtual bool intersect(Ray& r) const;
};
#endif
