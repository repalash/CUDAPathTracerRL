#include "camera.h"
#include <math.h>
#include <iostream>


Camera::Camera(const Vector3D& _pos, const Vector3D& _target, const Vector3D& _up, float _fovy, int _width, int _height) : 
position(_pos), target(_target), up(_up), fovy(_fovy), width(_width), height(_height)
{
	up.normalize();

	line_of_sight = target - position;

	steps = 0;

	//Calculate the camera basis vectors
	//Camera looks down the -w axis
	w = -line_of_sight;
	w.normalize();
	u = crossProduct(up, w);
	u.normalize();
	v = crossProduct(w, u);
	v.normalize();

	bitmap  = new unsigned char[width * height * 3 * sizeof(unsigned char)]; //RGB
	focalHeight = 1.0; //Let's keep this fixed to 1.0
	aspect = float(width)/float(height);
	focalWidth = focalHeight * aspect; //Height * Aspect ratio
	focalDistance = focalHeight/(2.0f * tan(fovy * M_PI/(180.0f * 2.0f))); //More the fovy, close is focal plane
}

Camera::~Camera()
{
	delete []bitmap;
}

//Get direction of viewing ray from pixel coordinates (i, j)
const Vector3D Camera::get_ray_direction(const float i, const float j) const
{
	Vector3D dir(0.0, 0.0, 0.0);
	dir += -w * focalDistance;
	float xw = (float) (aspect * (i - width / 2.0 + 0.5) / width);
	float yw = (float) ((j - height / 2.0 + 0.5) / height);
	dir += u * xw;
	dir += v * yw;

	dir.normalize();
	return dir;
}

void Camera::drawPixel(int i, int j, Color c)
{
	int index = (i + j*width)*3;
	bitmap[index + 0] = (unsigned char) ((bitmap[index + 0] * steps + 255 * c.r) / (steps + 1));
	bitmap[index + 1] = (unsigned char) ((bitmap[index + 1] * steps + 255 * c.g) / (steps + 1));
	bitmap[index + 2] = (unsigned char) ((bitmap[index + 2] * steps + 255 * c.b) / (steps + 1));
}
