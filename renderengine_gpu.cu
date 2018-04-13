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

RenderEngine_GPU::RenderEngine_GPU(World *_world, Camera *_camera) : RenderEngine(_world, _camera), wor(_world), cam(_camera) {
    //init vars
    cudaMalloc(reinterpret_cast<void**>(&bitmap_gpu), cam.size.y * cam.size.x * 3 * sizeof(unsigned char));
    cudaMalloc(reinterpret_cast<void**>(&random_texture_device), cam.size.y * cam.size.x * sizeof(int));
    random_texture = (int*)malloc(cam.size.y * cam.size.x * sizeof(int));
    //Init random texture.
    srand(static_cast<unsigned int>(clock()));
    for(int j = 0; j<cam.size.y * cam.size.x; j++){
        random_texture[j] = rand();
    }
    cudaMemcpy(random_texture_device, random_texture, cam.size.y * cam.size.x * sizeof(int), cudaMemcpyHostToDevice);

    //DO copy all variables
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
    Main_Render_Kernel << < blockspergrid, threadsperblock >> >(i, bitmap_gpu, cam, wor, steps, random_texture_device, clock());
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
        return false;
    }
    return false;
}

RenderEngine_GPU::~RenderEngine_GPU() {
    //Free variables
    cudaFree(bitmap_gpu);
    cudaFree(random_texture_device);
    free(random_texture);
}

__device__ float3 computeColor(Ray_GPU ray, int &seed, World_GPU wor) {
    float3 c = AMBIENT_COLOR;

    unsigned char sphere = wor.intersectRay(ray);

    for (unsigned char i = 0; i < MAX_LEVEL; i++){
        if(sphere^255) {
            c = c*wor.spheres[sphere].col;
            SPHERE_MATERIAL sp_mat = wor.spheres[sphere].material;
            if(sp_mat == LIGHT){
                //light
                break;
            }else if(sp_mat == DIELECTRIC){
                //dielectric
                float eta = wor.spheres[sphere].param;
                float cosTheta = dot(ray.dir, ray.normal);
                bool isInside = cosTheta > 0;
                float nc=1, nnt=isInside?eta/nc:nc/eta;
                float cos2t = 1-nnt*nnt*(1-cosTheta*cosTheta);
                if (cos2t<0){ //TIR
                    ray.dir = normalize(ray.dir - 2 * ray.normal * cosTheta);
                }else{
                    cosTheta = -fabs(cosTheta);
                    float3 refr_dir = normalize(ray.dir * nnt - ray.normal*((isInside?-1:1)*(cosTheta*nnt+sqrt(cos2t))));

                    float a=eta-nc, b=eta+nc, R0=a*a/(b*b), c1 = 1-(isInside?dot(refr_dir, ray.normal):-cosTheta);
                    float Re=R0+(1-R0)*c1*c1*c1*c1*c1,Tr=1-Re,P=.25f+.5f*Re,RP=Re/P,TP=Tr/(1-P);
                    if (Random_GPU(seed) < P) {
                        c = c * RP;
                        ray.dir = normalize(ray.dir - 2 * ray.normal * cosTheta);
                    }
                    else{
                        c = c * TP;
                        ray.dir = refr_dir;
                    }
                }
            }else if(sp_mat == GLOSSY){
                //glossy
                float cosTheta = dot(ray.dir, ray.normal);
                float n = wor.spheres[sphere].param;

                float phi=2*M_PI*Random_GPU(seed), cosAlpha=pow(Random_GPU(seed), 1.f/(n+1)), sineAlpha = sqrt(1-cosAlpha*cosAlpha);
                float rotAngle = 2*(acos(-cosTheta) + acos(cosAlpha) - M_PI/2);

                float3 w = normalize(ray.dir - 2 * ray.normal * cosTheta);
                float3 u = normalize(cross((fabs(w.x)>.1?make_float3(0,1,0):make_float3(1,0,0)),w));
                float3 v = cross(w,u);

                float3 dDirection = u*cos(phi)*sineAlpha + v*sin(phi)*sineAlpha + w*cosAlpha;

                if(dot(dDirection,ray.normal)<0) {
                    float3 k = normalize(cross(w, ray.normal));
                    dDirection = cos(rotAngle) * dDirection + sin(rotAngle) * cross(k, dDirection);
                }
                ray.dir = dDirection;
            }else if(sp_mat == REFLECTIVE && Random_GPU(seed) < wor.spheres[sphere].param){
                float cosTheta = dot(ray.dir, ray.normal);
                ray.dir = normalize(ray.dir - 2 * ray.normal * cosTheta);
            }else {
                //diffuse
                float alpha=2*M_PI* Random_GPU(seed),z= Random_GPU(seed), sineTheta = sqrtf(1-z);
                float3 w = ray.normal;
                float3 u = normalize(cross((fabs(w.x)>.1?make_float3(0,1,0):make_float3(1,0,0)),w));
                float3 v = cross(w,u);
                ray.dir = u*cos(alpha)*sineTheta + v*sin(alpha)*sineTheta + w*sqrt(z);
            }
            sphere = wor.intersectRay(ray);
        }else{
//            c = BACKGROUND;
            break;
        }
        if(i^3 && length(c) < 0.07){
            break;
        }
    }
    return c;
}

__global__ void Main_Render_Kernel(int startI, unsigned char *bitmap, Camera_GPU cam, World_GPU wor, unsigned int steps,
                                   int* rand_tex, int clk) { //j->row, i->column
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
    float3 c = computeColor(ray, seed, wor);

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
        float f = 1.0f / (steps+1);
        bitmap[index + 0] = (unsigned char) ((bitmap[index + 0] * (f * steps) + 255 * c.x * f));
        bitmap[index + 1] = (unsigned char) ((bitmap[index + 1] * (f * steps) + 255 * c.y * f));
        bitmap[index + 2] = (unsigned char) ((bitmap[index + 2] * (f * steps) + 255 * c.z * f));
    }
}
