//
// Created by Palash on 28-04-2018.
//

#ifndef PATHTRACER_CUDA_PATHTRACER_PARAMS_H
#define PATHTRACER_CUDA_PATHTRACER_PARAMS_H

#define SAMPLE 8
#define MAX_LEVEL 12
#define MAX_THREADS_IN_BLOCK 512
#define RAND_TEX_SIZE 1024
#define COLUMNS_IN_ONCE 320
#define AMBIENT_COLOR make_float3(1,1,1)
#define EPSILON 0.01f
#define INF 1<<24
#define MAX_COORD 15
#define BACKGROUND make_float3(0,0,0)
#define WARP_SIZE 32
#define ALPHA 0.0001f
#define ENABLE_RL 1
#define ENABLE_GPU 1


#endif //PATHTRACER_CUDA_PATHTRACER_PARAMS_H
