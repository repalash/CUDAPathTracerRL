#ifndef _LIGHTSOURCE_H_
#define _LIGHTSOURCE_H_

#include "color.h"
#include "vector3D.h"

class LightSource
{	
protected:
	World *world;
	Color intensity;
public:
	LightSource(World *_world, const Color _intensity):
		world(_world), intensity(_intensity)   {}
	virtual Vector3D getPosition() const = 0;
	Color getIntensity() const {return intensity; }

};
#endif
