//
// Created by Palash on 11-04-2018.
//

#include <ctime>
#include "renderengine_gpu.h"
#include "cudaHeaders.h"
#include "world_gpu.h"
#include "camera_gpu.h"
#include "ray_gpu.h"
#include "curand_kernel.h"
#include "material_functions.h"

RenderEngine_GPU::RenderEngine_GPU(World *_world, Camera *_camera) : RenderEngine(_world, _camera), wor(_world), cam(_camera) {
    //init vars
    cudaMalloc(reinterpret_cast<void**>(&bitmap_gpu), cam.size.y * cam.size.x * 3 * sizeof(unsigned char));
    cudaMalloc(reinterpret_cast<void**>(&random_texture_device), cam.size.y * cam.size.x * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&q_table_device), MAX_COORD * MAX_COORD * MAX_COORD * 8 * sizeof(float));
    random_texture = (int*)malloc(cam.size.y * cam.size.x * sizeof(int));
    q_table = (QNode*)malloc(MAX_COORD * MAX_COORD * MAX_COORD * 8 * sizeof(QNode));
    //Init random texture.
    srand(static_cast<unsigned int>(clock()));
    for(unsigned int j = 0; j<cam.size.y * cam.size.x; j++){
        random_texture[j] = rand();
    }
    for(int j = 0; j<MAX_COORD * MAX_COORD * MAX_COORD * 8; j++){
        for(int k=0; k<8; k++) {
            q_table[j].v[k] = 0.f;//.1f * rand() / RAND_MAX;
        }
        q_table[j].max = 0.f;
    }

    //DO copy all variables
    cudaMemcpy(random_texture_device, random_texture, cam.size.y * cam.size.x * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(q_table_device, q_table, MAX_COORD * MAX_COORD * MAX_COORD * 8 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bitmap_gpu, camera->getBitmap(), cam.size.y * cam.size.x * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
}

bool RenderEngine_GPU::renderLoop() {

    static int i = 0;
    static int steps = 0;

    cudaEvent_t begin, begin_kernel, stop_kernel, stop;
    cudaEventCreate(&begin);
    cudaEventCreate(&begin_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&stop);

    cudaEventRecord(begin);

    //Init random texture.
    srand(static_cast<unsigned int>(clock()));
    for(unsigned int j = 0; j<cam.size.y * cam.size.x; j++){
        random_texture[j] = rand();
    }
    cudaMemcpy(random_texture_device, random_texture, cam.size.y * cam.size.x * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsperblock(SAMPLE,SAMPLE,MAX_THREADS_IN_BLOCK/(SAMPLE*SAMPLE));
    dim3 blockspergrid(cam.size.y * COLUMNS_IN_ONCE/threadsperblock.z);

    cudaEventRecord(begin_kernel);
    Main_Render_Kernel << < blockspergrid, threadsperblock >> >(i, bitmap_gpu, cam, wor, steps, random_texture_device, clock(), q_table_device);
    cudaEventRecord(stop_kernel);
    gpuErrchk(cudaPeekAtLastError());

    //Copy all variables back
    cudaMemcpy(camera->getBitmap(), bitmap_gpu, cam.size.y * cam.size.x * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop_kernel);
    cudaEventSynchronize(stop);

    float kernelTime, totalTime; // Initialize elapsedTime;
    cudaEventElapsedTime(&kernelTime, begin_kernel, stop_kernel);
    cudaEventElapsedTime(&totalTime, begin, stop);

    if( (i+=COLUMNS_IN_ONCE) == camera->getWidth())
    {
        i = 0;
        steps++;
        printf("GPU Time: %fms, %fms, steps: %d\n", kernelTime, totalTime -kernelTime, steps);
        camera->incSteps();
//        std::cout<<"Samples Done: "<<camera->getSteps()*SAMPLE*SAMPLE<<std::endl;
        return steps >= 20;
    }
    return false;
}

RenderEngine_GPU::~RenderEngine_GPU() {
    //Free variables
    cudaFree(bitmap_gpu);
    cudaFree(random_texture_device);
    cudaFree(q_table_device);
    free(random_texture);
}

__global__ void Main_Render_Kernel(int startI, unsigned char *bitmap, Camera_GPU cam, World_GPU wor, unsigned int steps,
                                   int* rand_tex, int clk, QNode* q_table) { //j->row, i->column
    // <8,8,12>
    unsigned int p = threadIdx.x;
    unsigned int q = threadIdx.y;

    unsigned int j = (blockIdx.x * blockDim.z + threadIdx.z);
    unsigned int i = startI + j/cam.size.y;
    j %= cam.size.y;

    int seed = 341*q + 253 * p * 8 + ( rand_tex[(i + j*cam.size.x)%(cam.size.x * cam.size.y)]) + 349*steps + clk;
    float _i = i + 1.2f * (p + Random_GPU(seed)) / SAMPLE;
    float _j = j + 1.2f * (q + Random_GPU(seed)) / SAMPLE;

    //Initial Ray direction
    float xw = (1.0f*cam.size.x/cam.size.y * (_i - cam.size.x / 2.0f + 0.5f) / cam.size.x);
    float yw = ((_j - cam.size.y / 2.0f + 0.5f) / cam.size.y);
    float3 dir = normalize(cam.u * xw + cam.v * yw - cam.w * 1.207107f);

    //Create ray
    Ray_GPU ray(cam.pos, dir);
    float3 c = computeColor(ray, seed, wor, q_table, steps);

    c = warpReduceSumTriple(c);
    __shared__ float3 val[MAX_THREADS_IN_BLOCK/(SAMPLE*SAMPLE)];
    val[threadIdx.z] = make_float3(1,0,0);
    __syncthreads();
    if(p==SAMPLE-1 && q==SAMPLE-1)
        val[threadIdx.z] = c;
    __syncthreads();

    if(p==0 && q==0){
        c = c+val[threadIdx.z];
        c = clamp(c/(SAMPLE*SAMPLE), 0, 1);
        int index = (i + j*cam.size.x)*3;
//        steps = 0;
        float f = 1.0f / (steps+1);
//        if(length(c)>0.01)
        {
            bitmap[index + 0] = (unsigned char) ((bitmap[index + 0] * (f * steps) + 255 * c.x * f));
            bitmap[index + 1] = (unsigned char) ((bitmap[index + 1] * (f * steps) + 255 * c.y * f));
            bitmap[index + 2] = (unsigned char) ((bitmap[index + 2] * (f * steps) + 255 * c.z * f));
        }
    }
}
