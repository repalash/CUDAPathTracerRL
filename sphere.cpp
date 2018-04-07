//sphere.cpp

#include "sphere.h"
#include "lightsource.h"

bool Sphere::intersect(Ray& r) const
{
	Vector3D centerVector = r.getOrigin() - position;
	double a = 1.0;
	double b = 2*dotProduct(r.getDirection(), centerVector);
	double c = dotProduct(centerVector, centerVector) - radius*radius;
	double discriminant = b*b - 4.0*a*c;

	//now check if discriminant is positive or zero, then only we have an intersection!
	if(discriminant >=0.0)
	{
		if(discriminant == 0)
		{
			double t = -b/(2.0*a);
			Vector3D p = r.getOrigin() + t*r.getDirection();
			r.setParameter(t, this, p - position);
			return true;
		}
		else
		{
			//Calculate both roots
			double D = sqrt(discriminant);
			double t1 = (-b +D)/(2.0*a);
			double t2 = (-b -D)/(2.0*a);

			Vector3D p = r.getOrigin() + t1*r.getDirection();
			bool b1 = r.setParameter(t1, this, p - position);
			p = r.getOrigin() + t2*r.getDirection();
			bool b2 = r.setParameter(t2, this, p - position);
			return b1||b2;
		}
	}
	return false;
}


