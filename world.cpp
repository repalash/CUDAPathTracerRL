#include <iostream>
#include "world.h"

using namespace std;

float World::firstIntersection(Ray& ray)
{
	for(int i=0; i<objectList.size(); i++)
		objectList[i]->intersect(ray);
	return ray.getParameter();
}

Color World::shade_ray(Ray& ray)
{
	if(ray.getLevel() > MAX_LEVEL) return Color(ambient);
	firstIntersection(ray);
	if(ray.didHit()) {
		Color c = (ray.intersected())->shade(ray);
			return c;
	}
	return background;
}

void World::addLight(LightSource* ls) {
	Material *m = new Material(this); m->ka = 1; m->color = Color(ls->getIntensity());
	Sphere *sphere = new Sphere(ls->getPosition(), 8, m);
	sphere->setLightSource(ls);
	lightSourceList.push_back(ls);
	addObject(sphere);
}
