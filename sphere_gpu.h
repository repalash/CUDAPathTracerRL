//
// Created by Palash on 12-04-2018.
//

#ifndef PATHTRACER_CUDA_SPHERE_GPU_H
#define PATHTRACER_CUDA_SPHERE_GPU_H

#include <vector_types.h>
#include "sphere.h"
#include "cudaHeaders.h"
#include "cuda_utils.h"
#include "ray_gpu.h"

enum SPHERE_MATERIAL { DIFFUSE, LIGHT, DIELECTRIC };

struct Sphere_GPU {
    float rad;
    float3 pos, col;
    float param;
    SPHERE_MATERIAL material;
    __host__ Sphere_GPU(){}
    __host__ Sphere_GPU(Sphere *s) {
        rad = static_cast<float>(s->getRadius());
        Vector3D_To_float3(s->getPosition(), &pos);
        Color_To_float3(s->getMaterial()->color, &col);
        material = DIFFUSE;
        if(s->isLightSource()){
            material = LIGHT;
            Color_To_float3(s->getLightSource()->getIntensity(), &col);
        }
        if(s->getMaterial()->kt>0){
            material = DIELECTRIC;
            param = static_cast<float>(s->getMaterial()->eta);
        }
    }
    __device__ float intersect(const Ray_GPU &r) const {
        float3 op = pos - r.orig;
        float t;
        float b = dot(op, r.dir);
        float disc = b*b - dot(op, op) + rad*rad;
        if (disc<0) return 0;
        else disc = sqrtf(disc);
        t = (t = b - disc)>EPSILON ? t : ((t = b + disc)>EPSILON ? t : 0);
        return t;
    }
};


#endif //PATHTRACER_CUDA_SPHERE_GPU_H
