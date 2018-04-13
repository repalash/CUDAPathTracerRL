//
// Created by Palash on 12-04-2018.
//

#ifndef PATHTRACER_CUDA_WORLD_GPU_H
#define PATHTRACER_CUDA_WORLD_GPU_H

#include "world.h"
#include "sphere_gpu.h"

struct World_GPU{
    Sphere_GPU spheres[20];
    unsigned char n = 0;
    __host__ World_GPU(World *wor){
        for(int i=0; i<wor->getObjectList().size(); i++){
            spheres[i] = Sphere_GPU((Sphere*)wor->getObjectList()[i]);
            n++;
        }
    }

    __device__ unsigned char intersectRay(Ray_GPU &ray) {
        float t = INF;
        unsigned char sph = 255;
        for (unsigned char i = 0; i < n; i++){
            float nt = spheres[i].intersect(ray);
            if(nt>0&&nt<t){
                t = nt;
                sph = i;
            }
        }
        if(sph<n) {
            float3 er = ray.orig + t * ray.dir;
            ray.orig = er;
            ray.normal = normalize(er - spheres[sph].pos);
        }
        return sph;
    }
};

#endif //PATHTRACER_CUDA_WORLD_GPU_H
