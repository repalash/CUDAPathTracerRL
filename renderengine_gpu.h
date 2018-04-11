//
// Created by Palash on 11-04-2018.
//

#ifndef _RENDERENGINE_GPU_H_
#define _RENDERENGINE_GPU_H_

#include "renderengine.h"
#include "cudaHeaders.h"
#include "world_gpu.h"
#include "camera_gpu.h"

class RenderEngine_GPU: public RenderEngine{

public:
    RenderEngine_GPU(World *_world, Camera *_camera);
    bool renderLoop();
};

__global__ void Main_Render_Kernel(int startI, unsigned char* bitmap, Camera_GPU cam, World_GPU wor, unsigned int steps,
                                   unsigned int mrand);
__device__ float3 computeColor(Ray_GPU i, unsigned int &j, World_GPU wor);


#endif //PATHTRACER_CUDA_RENDERENGINE_GPU_H
