//vector3D.h
#ifndef _VECTOR_H_
#define _VECTOR_H_
#include<math.h>

#if !defined(M_PI)
#   define M_PI 3.1415926535897932384626433832795f
#endif  // !defined(M_PI)

class Vector3D
{
public:
	Vector3D() {}
	Vector3D(double e0, double e1, double e2);
	Vector3D(const Vector3D &v) {*this = v;}

	double X() const{ return e[0];}
	double Y() const{ return e[1];}
	double Z() const{ return e[2];}

	void X(double x) {e[0] = x;}
	void Y(double y) {e[1] = y;}
	void Z(double z) {e[2] = z;}

	//define operators
	const Vector3D& operator+() const;
	Vector3D operator-() const;
	double operator[](int i) const {return e[i];}
	double& operator[](int i) {return e[i];}

	friend bool operator==(const Vector3D& v1, const Vector3D& v2);
	friend bool operator!=(const Vector3D& v1, const Vector3D& v2);
	friend Vector3D operator+(const Vector3D& v1, const Vector3D& v2);
	friend Vector3D operator-(const Vector3D& v1, const Vector3D& v2);
	friend Vector3D operator/(const Vector3D& v, double scalar);
	friend Vector3D operator*(const Vector3D& v, double scalar);
	friend Vector3D operator*(double scalar, const Vector3D& v);
	Vector3D& operator+=(const Vector3D &v);
	Vector3D& operator-=(const Vector3D &v);
	Vector3D& operator*=(double scalar);
	Vector3D& operator/=(double scalar);

	//Vector3D functions
	double length() const;
	double squaredlength() const;
	void normalize();
	friend Vector3D unitVector(const Vector3D& v);
	friend Vector3D crossProduct(const Vector3D& v1, const Vector3D& v2);
	friend double dotProduct(const Vector3D& v1, const Vector3D& v2);
	friend double tripleProduct(const Vector3D& v1,const Vector3D& v2,const Vector3D& v3);

	//data member
	double e[3];
};
#endif
