//
// Created by Palash on 28-04-2018.
//

#ifndef PATHTRACER_CUDA_MATERIAL_FUNCTIONS_H
#define PATHTRACER_CUDA_MATERIAL_FUNCTIONS_H

#include "cutil_math.h"
#include "world_gpu.h"
#include "renderengine_gpu.h"
#include "rl_helper_functions.h"

__device__ __inline__ inline float3 shadeDielectric(Ray_GPU &ray, int &seed, World_GPU &wor, float3 &c, unsigned char &sphere) {
    float eta = wor.spheres[sphere].param;
    float cosTheta = dot(ray.dir, ray.normal);
    bool isInside = cosTheta > 0;
    float nc = 1, nnt = isInside ? eta / nc : nc / eta;
    float cos2t = 1 - nnt * nnt * (1 - cosTheta * cosTheta);
    if (cos2t < 0) { //TIR
        return normalize(ray.dir - 2 * ray.normal * cosTheta);
    } else {
        cosTheta = -fabs(cosTheta);
        float3 refr_dir = normalize(
                ray.dir * nnt - ray.normal * ((isInside ? -1 : 1) * (cosTheta * nnt + sqrt(cos2t))));

        float a = eta - nc, b = eta + nc, R0 = a * a / (b * b), c1 =
                1 - (isInside ? dot(refr_dir, ray.normal) : -cosTheta);
        float Re = R0 + (1 - R0) * c1 * c1 * c1 * c1 * c1, Tr = 1 - Re, P = .25f + .5f * Re, RP =
                Re / P, TP = Tr / (1 - P);
        if (Random_GPU(seed) < P) {
            c = c * RP;
            return normalize(ray.dir - 2 * ray.normal * cosTheta);
        } else {
            c = c * TP;
            return refr_dir;
        }
    }
}

__device__ __inline__ inline float3 shadeGlossy(Ray_GPU &ray, int &seed, World_GPU &wor, float3 &c, unsigned char &sphere){
    float cosTheta = dot(ray.dir, ray.normal);
    float n = wor.spheres[sphere].param;

    float phi = 2 * M_PI * Random_GPU(seed), cosAlpha = pow(Random_GPU(seed),
                                                            1.f / (n + 1)), sineAlpha = sqrt(
            1 - cosAlpha * cosAlpha);
    float rotAngle = 2 * (acos(-cosTheta) + acos(cosAlpha) - M_PI / 2);

    float3 w = normalize(ray.dir - 2 * ray.normal * cosTheta);
    float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
    float3 v = cross(w, u);

    float3 dDirection = u * cos(phi) * sineAlpha + v * sin(phi) * sineAlpha + w * cosAlpha;

    if (dot(dDirection, ray.normal) < 0) {
        float3 k = normalize(cross(w, ray.normal));
        dDirection = cos(rotAngle) * dDirection + sin(rotAngle) * cross(k, dDirection);
    }
    return dDirection;
}

__device__ __inline__ inline float3 shadeReflective(Ray_GPU &ray, int &seed, World_GPU &wor, float3 &c, unsigned char &sphere){
    float cosTheta = dot(ray.dir, ray.normal);
    return normalize(ray.dir - 2 * ray.normal * cosTheta);
}

__device__ __inline__ inline float3 shadeDiffuse(Ray_GPU &ray, int &seed) {
    float alpha = 2 * M_PI * Random_GPU(seed), z = Random_GPU(seed), sineTheta = sqrtf(1 - z);
    float3 w = ray.normal;
    float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
    float3 v = cross(w, u);
    return normalize(u * cos(alpha) * sineTheta + v * sin(alpha) * sineTheta + w * sqrt(z));
}

#define DEBUG 0

__device__ float3 computeColor(Ray_GPU &ray, int &seed, World_GPU &wor, QNode* &q_table, unsigned int &steps) {
    float3 c = AMBIENT_COLOR, c_q_table = make_float3(0);

#if ENABLE_RL
    unsigned int q_index = getQIndex(ray.orig), last_index=0;
    unsigned char dir_quad = getDirectionOctant(ray.dir), last_dir_quad=0;
#endif
    unsigned char sphere = wor.intersectRay(ray);
    bool isL = false;
    for (unsigned char i = 0; i < MAX_LEVEL; i++){
#if ENABLE_RL
        last_index = q_index;
//        last_dir_quad = dir_quad;
        q_index = getQIndex(ray.orig);
        last_dir_quad = getDirectionOctant(ray.dir);
//        c = make_float3((floor(ray.orig.x) + MAX_COORD)/(MAX_COORD*2), (floor(ray.orig.y) + MAX_COORD)/(MAX_COORD*2), (floor(ray.orig.z) + MAX_COORD)/(MAX_COORD*2));
//        if(DEBUG && !i) c_q_table = make_float3(q_table[q_index].v[3]);
        if(DEBUG && !i) c_q_table = (ray.dir);
#endif
//        break;
        if(sphere^255) {
            c = c*wor.spheres[sphere].col;
            SPHERE_MATERIAL sp_mat = wor.spheres[sphere].material;
            if(sp_mat == LIGHT){
                //light
                isL = true;
#if ENABLE_RL
                updateQTable(q_table, last_index, last_dir_quad, clamp01(length(wor.spheres[sphere].col)));
#endif
                break;
            }else {
#if ENABLE_RL
                updateQTable(q_table, last_index, last_dir_quad, clamp01(q_table[q_index].max * 0.8f));
#endif
                if (sp_mat == DIELECTRIC) {
                    //dielectric
                    ray.dir = shadeDielectric(ray, seed, wor, c, sphere);
                    sphere = wor.intersectRay(ray);
                } else if (sp_mat == GLOSSY) {
                    //glossy
                    ray.dir = shadeGlossy(ray, seed, wor, c, sphere);
                    sphere = wor.intersectRay(ray);
                } else if (sp_mat == REFLECTIVE && Random_GPU(seed) < wor.spheres[sphere].param) {
                    ray.dir = shadeReflective(ray, seed, wor, c, sphere);
                    sphere = wor.intersectRay(ray);
                } else {
                    //diffuse
#if ENABLE_RL
                    float3 direction = make_float3(0);// = shadeDiffuse(ray, seed);
                    unsigned char t_index;// = getDirectionOctant(direction);
                    QNode q = q_table[q_index];
                    if(steps>3) for(int li=0; li<16; li++) {
                        if(DEBUG && !i) c_q_table = make_float3(0.f);
                        direction = shadeDiffuse(ray, seed);
                        t_index = getDirectionOctant(direction);
                        if (q.v[t_index] > 0.75 * q.max) {
                            if(DEBUG && !i) c_q_table = make_float3(t_index/7.f);
                            break;
                        }
                    }else direction = shadeDiffuse(ray, seed);

//                    if(DEBUG && !i) for(int li=0;li<8;li++){
//                        if(q.v[li] < 0.95*q.max)
//                            c_q_table = make_float3(li/7.f);
//                    }
                    ray.dir = direction;
#else
                    ray.dir = shadeDiffuse(ray, seed);
#endif
                    sphere = wor.intersectRay(ray);
                }
            }
        }else{
//            c = BACKGROUND;
            break;
        }
//        if(length(c) < 0.07)
//            break;
    }
    return DEBUG?c_q_table:isL?c:make_float3(0);
}

#endif //PATHTRACER_CUDA_MATERIAL_FUNCTIONS_H
