//
// Created by Palash on 11-04-2018.
//

#include "renderengine_gpu.h"
#include "cudaHeaders.h"

#define SAMPLE 8
#define MAX_THREADS_IN_BLOCK 768
#define COLUMNS_IN_ONCE 360
#define IMAGE_HEIGHT 240
#define IMAGE_WIDTH 360
#define AMBIENT_COLOR make_float3(1,1,1)
#define EPSILON 0.0001f
#define INF 1<<24
#define BACKGROUND make_float3(0,0,0)
#define WARP_SIZE 32

__device__ float xorshf96_gpu(dim3* s) {
    s->x = (s->x * 256634895 + 2343786)%837234723;
    s->y = (s->x + s->y * 23565645 + 456893)%234235723;
    s->z = (s->y + s->z * 34578345 + 7547556)%534546723;
    return 1.0f*(s->z)/534546723;
}
__host__ void Vector3D_To_float3(Vector3D a, float3 *b){
    b->x = static_cast<float>(a.X());
    b->y = static_cast<float>(a.Y());
    b->z = static_cast<float>(a.Z());
};
__host__ void Color_To_float3(Color a, float3 *b){
    b->x = static_cast<float>(a.r);
    b->y = static_cast<float>(a.g);
    b->z = static_cast<float>(a.b);
};

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
struct Sphere_GPU {
    float rad;
    float3 pos, col;
    int light;
    __host__ Sphere_GPU(){}
    __host__ Sphere_GPU(Sphere *s) {
        rad = static_cast<float>(s->getRadius());
        Vector3D_To_float3(s->getPosition(), &pos);
        Color_To_float3(s->getMaterial()->color, &col);
        if((light = s->isLightSource()?1:0)){
            Color_To_float3(s->getLightSource()->getIntensity(), &col);
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

struct Camera_GPU{
    float3 u;
    float3 v;
    float3 w;
    float3 pos;
    __host__ Camera_GPU(Camera *cam){
        Vector3D_To_float3(cam->getU(), &u);
        Vector3D_To_float3(cam->getV(), &v);
        Vector3D_To_float3(cam->getW(), &w);
        Vector3D_To_float3(cam->get_position(), &pos);
    }
};
struct World_GPU{
    Sphere_GPU spheres[10];
    unsigned char n = 0;
    __host__ World_GPU(World *wor){
        for(int i=0; i<wor->getObjectList().size(); i++){
            spheres[i] = Sphere_GPU((Sphere*)wor->getObjectList()[i]);
            n++;
        }
    }

    __device__ unsigned char intersectRay(Ray_GPU &ray) {
        float t = INF;
        unsigned char sph = n;
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

__global__ void Main_Render_Kernel(int startI, unsigned char* bitmap, Camera_GPU cam, World_GPU wor, unsigned int steps); //j->row, i->column

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
    Main_Render_Kernel << < blockspergrid, threadsperblock >> >(i, bitmap_gpu, cam, wor, steps);
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
        return steps>10;
    }
    return false;
}



__device__ float3 computeColor(Ray_GPU i, dim3 *j, World_GPU wor);

__device__ float3 warpReduceSumTriple(float3 val) {
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val.x += __shfl_xor(val.x, offset);
        val.y += __shfl_xor(val.y, offset);
        val.z += __shfl_xor(val.z, offset);
    }
    return val;
}

__global__ void Main_Render_Kernel(int startI, unsigned char* bitmap, Camera_GPU cam, World_GPU wor, unsigned int steps){ //j->row, i->column
	// <8,8,12>

	unsigned int p = threadIdx.x;
	unsigned int q = threadIdx.y;

	unsigned int j = (blockIdx.x * blockDim.z + threadIdx.z);
	unsigned int i = startI + j/IMAGE_HEIGHT;
	j %= IMAGE_HEIGHT;

    dim3 seed(123456789+866*p+359*j+steps*1234, 362436069+i*2347+q*213+steps*5341,521288629+235*i*j+steps*9834);
	float _i = i + (p + xorshf96_gpu(&seed)) / SAMPLE;
	float _j = j + (q + xorshf96_gpu(&seed)) / SAMPLE;


	//Initial Ray direction
	float3 dir = make_float3(0,0,0);
    dir += -cam.w * 1.207107f;
    float xw = (1.0f*IMAGE_WIDTH/IMAGE_HEIGHT * (i - IMAGE_WIDTH / 2.0f + 0.5f) / IMAGE_WIDTH);
    float yw = ((j - IMAGE_HEIGHT / 2.0f + 0.5f) / IMAGE_HEIGHT);
    dir += cam.u * xw;
    dir += cam.v * yw;
    dir = normalize(dir);

    //Create ray
    Ray_GPU ray(cam.pos, dir);
    float3 c = computeColor(ray, &seed, wor);

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

__device__ float3 computeColor(Ray_GPU ray, dim3 *seed, World_GPU wor) {
    float3 c = AMBIENT_COLOR;

    unsigned char sphere = wor.intersectRay(ray);
    int loop_end = 0;
    for (unsigned char i = 0; i < MAX_LEVEL; i++){
        if(loop_end){
            continue;
        }
        else if(sphere<wor.n) {
            c = c*wor.spheres[sphere].col;
            if(!wor.spheres[sphere].light) {
                float alpha=2*M_PI*xorshf96_gpu(seed),z=xorshf96_gpu(seed), sineTheta = sqrtf(1-z);
                float3 w = ray.normal;
                float3 u = normalize(cross((fabs(w.x)>.1?make_float3(0,1,0):make_float3(1,0,0)),w));
                float3 v = cross(w,u);
                ray.dir = u*cos(alpha)*sineTheta + v*sin(alpha)*sineTheta + w*sqrt(z);
                sphere = wor.intersectRay(ray);
            }else {
                loop_end = 1;
            }
//        if(kg>0)
//            return finalColor*world->shade_ray(randomRay) *(kg * pow(dotProduct(rDirection, dDirection), n)) * dotProduct(w, dDirection); //Glossy
        }else{
            c = BACKGROUND;
            loop_end = true;
        }
    }
    return c;
}