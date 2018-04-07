#ifndef _RENDERENGINE_H_
#define _RENDERENGINE_H_

#include "world.h"
#include "camera.h"
#include "material.h"

class RenderEngine
{
private:
	World *world;
	Camera *camera;
	const Color trace(const float i, const float j);

public:
	RenderEngine(World *_world, Camera *_camera):
		world(_world), camera(_camera) {}
	bool renderLoop();
};
#endif
