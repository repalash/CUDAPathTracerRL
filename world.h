#ifndef _WORLD_H_
#define _WORLD_H_

#define MAX_LEVEL 4

#include <vector>
#include "object.h"
#include "lightsource.h"
#include "color.h"
#include "sphere.h"
#include "ray.h"

class World
{
private:
	std::vector<Object*> objectList;

	Color ambient;
	Color background; //Background color to shade rays that miss all objects

public:
	World():
		objectList(0), ambient(0), background(0)
	{}
	void setBackground(const Color& bk) { background = bk;}
	Color getBackground() { return background;}
	void setAmbient(const Color& amb) {ambient = amb;}
	Color getAmbient() {return ambient;}

	void addLight(LightSource* ls);
	void addObject(Object *obj)
	{
		objectList.push_back(obj);
	}
	float firstIntersection(Ray& ray);
	Color shade_ray(Ray& ray);
};

#endif
