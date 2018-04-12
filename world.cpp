#include <iostream>
#include "world.h"
#include "pointlightsource.h"

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

void World::addLight(PointLightSource *ls, float radius) {
	Material *m = new Material(this); m->ka = 1; m->color = Color(ls->getIntensity());
	Sphere *sphere = new Sphere(ls->getPosition(), radius, m);
	sphere->setLightSource(ls);
	addObject(sphere);
}

const vector<Object *> &World::getObjectList() const {
    return objectList;
}
