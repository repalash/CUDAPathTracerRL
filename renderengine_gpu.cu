//
// Created by Palash on 11-04-2018.
//

#include "renderengine_gpu.h"
#include "cudaHeaders.h"
#include "world_gpu.h"
#include "camera_gpu.h"
#include "ray_gpu.h"

RenderEngine_GPU::RenderEngine_GPU(World *_world, Camera *_camera) : RenderEngine(_world, _camera) {

}

bool RenderEngine_GPU::renderLoop() {

    static int i = 0;
    static int steps = 0;

    cudaEvent_t begin, begin_kernel, stop_kernel, stop;
    cudaEventCreate(&begin);
    cudaEventCreate(&begin_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&stop);

    //init vars
    unsigned char *bitmap_gpu;
    cudaMalloc(reinterpret_cast<void**>(&bitmap_gpu), IMAGE_HEIGHT * IMAGE_WIDTH * 3 * sizeof(unsigned char));
    Camera_GPU cam(camera);
    World_GPU wor(world);

    cudaEventRecord(begin);

    //DO copy all variables
    cudaMemcpy(bitmap_gpu, camera->getBitmap(), IMAGE_HEIGHT * IMAGE_WIDTH * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsperblock(SAMPLE,SAMPLE,MAX_THREADS_IN_BLOCK/(SAMPLE*SAMPLE));
    dim3 blockspergrid(IMAGE_HEIGHT * COLUMNS_IN_ONCE/threadsperblock.z);

    cudaEventRecord(begin_kernel);
    Main_Render_Kernel << < blockspergrid, threadsperblock >> >(i, bitmap_gpu, cam, wor, steps, rand());
    cudaEventRecord(stop_kernel);
    gpuErrchk(cudaPeekAtLastError());

    //Copy all variables back
    cudaMemcpy(camera->getBitmap(), bitmap_gpu, IMAGE_HEIGHT * IMAGE_WIDTH * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop_kernel);
    cudaEventSynchronize(stop);

    float kernelTime, totalTime; // Initialize elapsedTime;
    cudaEventElapsedTime(&kernelTime, begin_kernel, stop_kernel);
    cudaEventElapsedTime(&totalTime, begin, stop);

    printf("Time: %fms, %fms\n", kernelTime, totalTime);

    //Free variables
    cudaFree(bitmap_gpu);

    if( (i+=COLUMNS_IN_ONCE) == camera->getWidth())
    {
        i = 0;
        steps++;
        camera->incSteps();
//        std::cout<<"Samples Done: "<<camera->getSteps()*SAMPLE*SAMPLE<<std::endl;
        return false;
    }
    return false;
}

__device__ float3 computeColor(Ray_GPU ray, unsigned int &seed, World_GPU wor) {
    float3 c = AMBIENT_COLOR;

    unsigned char sphere = wor.intersectRay(ray);
    int loop_end = 0;
    for (unsigned char i = 0; i < MAX_LEVEL; i++){
        if(loop_end){
            continue;
        }
        else if(sphere<wor.n) {
            c = c*wor.spheres[sphere].col;
            SPHERE_MATERIAL sp_mat = wor.spheres[sphere].material;
            if(sp_mat == LIGHT){
                //light
                loop_end = 1;
                continue;
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
            c = BACKGROUND;
            loop_end = true;
        }
    }
    return c;
}

__global__ void Main_Render_Kernel(int startI, unsigned char *bitmap, Camera_GPU cam, World_GPU wor, unsigned int steps,
                                   unsigned int mrand) { //j->row, i->column
    // <8,8,12>

    unsigned int p = threadIdx.x;
    unsigned int q = threadIdx.y;

    unsigned int j = (blockIdx.x * blockDim.z + threadIdx.z);
    unsigned int i = startI + j/IMAGE_HEIGHT;
    j %= IMAGE_HEIGHT;

    unsigned int seed = 12345678 + p*11234 + q*23145 + i*13456 + j*14567 + steps*5678 + mrand*49574;
    float _i = i + (p + Random_GPU(seed)) / SAMPLE;
    float _j = j + (q + Random_GPU(seed)) / SAMPLE;


    //Initial Ray direction
    float3 dir = make_float3(0,0,0);
    dir += -cam.w * 1.207107f;
    float xw = (1.0f*IMAGE_WIDTH/IMAGE_HEIGHT * (_i - IMAGE_WIDTH / 2.0f + 0.5f) / IMAGE_WIDTH);
    float yw = ((_j - IMAGE_HEIGHT / 2.0f + 0.5f) / IMAGE_HEIGHT);
    dir += cam.u * xw;
    dir += cam.v * yw;
    dir = normalize(dir);

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
        int index = (i + j*IMAGE_WIDTH)*3;
        bitmap[index + 0] = (unsigned char) ((bitmap[index + 0] * steps + 256  * c.x) / (steps + 1));
        bitmap[index + 1] = (unsigned char) ((bitmap[index + 1] * steps + 256 * c.y) / (steps + 1));
        bitmap[index + 2] = (unsigned char) ((bitmap[index + 2] * steps + 256  * c.z) / (steps + 1));
    }
}
