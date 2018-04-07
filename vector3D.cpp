//vector3D.cpp

#include "vector3D.h"
#include <assert.h>

Vector3D::Vector3D(double e0, double e1, double e2)
{
	e[0] = e0;
	e[1] = e1;
	e[2] = e2;
}

const Vector3D& Vector3D::operator+() const
{return *this;}

Vector3D Vector3D::operator-() const
{return Vector3D(-e[0], -e[1], -e[2]);}

bool operator==(const Vector3D& v1, const Vector3D& v2)
{
	if (v1.e[0] != v2.e[0]) return false;
	if (v1.e[1] != v2.e[1]) return false;
	if (v1.e[2] != v2.e[2]) return false;
	return true;
}

bool operator!=(const Vector3D& v1, const Vector3D& v2)
{
	return !(v1==v2);   
}

Vector3D operator+(const Vector3D& v1, const Vector3D& v2)
{
	return Vector3D(v1.e[0]+v2.e[0], v1.e[1]+v2.e[1], v1.e[2]+v2.e[2]);
}

Vector3D operator-(const Vector3D& v1, const Vector3D& v2)
{
	return Vector3D(v1.e[0]-v2.e[0], v1.e[1]-v2.e[1], v1.e[2]-v2.e[2]);   
}

Vector3D operator/(const Vector3D& v, double scalar)
{
	return Vector3D(v.e[0]/scalar, v.e[1]/scalar, v.e[2]/scalar);   
}

Vector3D operator*(const Vector3D& v, double scalar)
{
	return Vector3D(v.e[0]*scalar, v.e[1]*scalar, v.e[2]*scalar);       
}

Vector3D operator*(double scalar, const Vector3D& v)
{
	return Vector3D(v.e[0]*scalar, v.e[1]*scalar, v.e[2]*scalar);       
}

Vector3D& Vector3D::operator+=(const Vector3D &v)
{
	e[0] += v.e[0]; e[1] += v.e[1]; e[2] += v.e[2];
	return *this;
}

Vector3D& Vector3D::operator-=(const Vector3D &v)
{
	e[0] -= v.e[0]; e[1] -= v.e[1]; e[2] -= v.e[2];
	return *this;
}

Vector3D& Vector3D::operator*=(double scalar)
{
	e[0] *= scalar; e[1] *= scalar; e[2] *= scalar;
	return *this;
}

Vector3D& Vector3D::operator/=(double scalar)
{
	assert(scalar != 0);
	float inv = 1.f/scalar;
	e[0] *= inv; e[1] *= inv; e[2] *= inv;
	return *this;
}

double Vector3D::length() const
{ return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }

double Vector3D::squaredlength() const
{ return (e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }

void Vector3D::normalize()
{ *this = *this / (*this).length();}

Vector3D unitVector(const Vector3D& v)
{
	double length  = v.length();
	return v / length;
}

Vector3D crossProduct(const Vector3D& v1, const Vector3D& v2)
{
	Vector3D tmp;
	tmp.e[0] = v1.Y() * v2.Z() - v1.Z() * v2.Y();
	tmp.e[1] = v1.Z() * v2.X() - v1.X() * v2.Z();
	tmp.e[2] = v1.X() * v2.Y() - v1.Y() * v2.X();
	return tmp; 
}

double dotProduct(const Vector3D& v1, const Vector3D& v2)
{ return v1.X()*v2.X() + v1.Y()*v2.Y() + v1.Z()*v2.Z(); }

double tripleProduct(const Vector3D& v1,const Vector3D& v2,const Vector3D& v3)
{
	return dotProduct(( crossProduct(v1, v2)), v3);   
}
