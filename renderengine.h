#ifndef _RENDERENGINE_H_
#define _RENDERENGINE_H_

#include "world.h"
#include "camera.h"
#include "material.h"

class RenderEngine
{
protected:
	World *world;
	Camera *camera;

public:
	RenderEngine(World *_world, Camera *_camera):
		world(_world), camera(_camera) {}
	virtual bool renderLoop();
	virtual const Color trace(const float i, const float j);
};
#endif
