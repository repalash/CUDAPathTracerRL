//
// Created by Palash on 12-04-2018.
//

#ifndef PATHTRACER_CUDA_RAY_GPU_H
#define PATHTRACER_CUDA_RAY_GPU_H

#include <vector_types.h>

struct Ray_GPU {
    float3 orig; // ray origin
    float3 dir;  // ray direction
    float3 normal;
//    unsigned char level = 0;
    __device__ Ray_GPU(float3 o, float3 d) {
        orig = o;
        dir = d;
    }
};

#endif //PATHTRACER_CUDA_RAY_GPU_H
