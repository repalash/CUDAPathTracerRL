//
// Created by Palash on 12-04-2018.
//

#ifndef PATHTRACER_CUDA_CAMERA_GPU_H
#define PATHTRACER_CUDA_CAMERA_GPU_H

#include <vector_types.h>
#include "camera.h"
#include "cuda_utils.h"

struct Camera_GPU{
    float3 u;
    float3 v;
    float3 w;
    float3 pos;
    uint2 size;
    __host__ Camera_GPU(Camera *cam){
        Vector3D_To_float3(cam->getU(), &u);
        Vector3D_To_float3(cam->getV(), &v);
        Vector3D_To_float3(cam->getW(), &w);
        Vector3D_To_float3(cam->get_position(), &pos);
        size = {cam->getWidth(), cam->getHeight()};
    }
};

#endif //PATHTRACER_CUDA_CAMERA_GPU_H
