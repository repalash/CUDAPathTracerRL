//object.h
#ifndef _OBJECT_H_
#define _OBJECT_H_

#include "ray.h"
#include "vector3D.h"
#include "color.h"
#include "material.h"
#include "lightsource.h"

class Object
{
protected:
	Material *material;
	bool isSolid;
	LightSource* lightSource;

public:
	Object(Material *mat): material(mat) {
		lightSource = nullptr;
	}
	virtual bool intersect(Ray& ray) const = 0;
	virtual Color shade(const Ray& ray) const
	{
		return material->shade(ray, isSolid);
	}
	void setLightSource(LightSource *ls){ lightSource = ls; }
	bool isLightSource() const { return lightSource!=nullptr; };
	const LightSource* getLightSource() const { return lightSource; };
	Material *getMaterial() const {
		return material;
	}
};

#endif
