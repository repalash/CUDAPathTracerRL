#include <random>
#include "renderengine.h"
#include "cudaHeaders.h"
#include <cmath>
#include <iostream>
#include <ctime>

#define SAMPLE 8

const Color RenderEngine::trace(const float i, const float j)
{
	Vector3D ray_dir = camera->get_ray_direction(i, j);
	Ray ray(camera->get_position(), ray_dir);
	return world->shade_ray(ray);
}

bool RenderEngine::renderLoop()
{
	static int i = 0;
	static clock_t cl = clock();
#pragma omp parallel for schedule(dynamic, 5)
	for(int j = 0; j<camera->getHeight(); j++)
	{
		Color color(0);
		for(int p =0; p<SAMPLE; p++){
			for(int q=0; q<SAMPLE; q++){
				color = color + trace((const float) (i + (p + xorshf96()) / SAMPLE), (const float) (j + (q + xorshf96()) / SAMPLE));
			}
		}
		color = color / (SAMPLE*SAMPLE);
		color.clamp();
		camera->drawPixel(i, j, color);
	}
    printf("CPU Time: %fms, steps: %d\n", 1000 * (double)(clock() - cl) / CLOCKS_PER_SEC, i);

	if(++i == camera->getWidth())
	{
		i = 0;
		camera->incSteps();
//		std::cout<<"Samples Done: "<<camera->getSteps()*SAMPLE*SAMPLE<<std::endl;
		return false;
	}
	return false;
}

//#define SAMPLE 8
