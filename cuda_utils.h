//
// Created by Palash on 12-04-2018.
//

#ifndef PATHTRACER_CUDA_CUDA_UTILS_H
#define PATHTRACER_CUDA_CUDA_UTILS_H

#include "cudaHeaders.h"
#include "vector3D.h"
#include "color.h"

__host__ __inline__ inline void Vector3D_To_float3(Vector3D a, float3 *b) {
    b->x = static_cast<float>(a.X());
    b->y = static_cast<float>(a.Y());
    b->z = static_cast<float>(a.Z());
}

__host__ __inline__ inline void Color_To_float3(Color a, float3 *b) {
    b->x = static_cast<float>(a.r);
    b->y = static_cast<float>(a.g);
    b->z = static_cast<float>(a.b);
}

__device__ __inline__ inline float Random_GPU(unsigned int &s) {
    s = (s * 16843009 + 826366247);
    return 1.0f*((s>>16)&0x7fff)/0x7fff;
}

__device__ __inline__ inline float3 warpReduceSumTriple(float3 val) {
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val.x += __shfl_xor(val.x, offset);
        val.y += __shfl_xor(val.y, offset);
        val.z += __shfl_xor(val.z, offset);
    }
    return val;
}


#endif //PATHTRACER_CUDA_CUDA_UTILS_H
