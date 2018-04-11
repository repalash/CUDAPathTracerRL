//
// Created by Palash on 11-04-2018.
//

#ifndef _RENDERENGINE_GPU_H_
#define _RENDERENGINE_GPU_H_

#include "renderengine.h"

class RenderEngine_GPU: public RenderEngine{

public:
    RenderEngine_GPU(World *_world, Camera *_camera);
    bool renderLoop();
};

#endif //PATHTRACER_CUDA_RENDERENGINE_GPU_H
