//
// Created by Palash on 19-02-2018.
//

#ifndef CUDA_BASE_CUDAHEADERS_H
#define CUDA_BASE_CUDAHEADERS_H

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include "cutil_math.h"

#define SAMPLE 8
#define MAX_THREADS_IN_BLOCK 512
#define RAND_TEX_SIZE 1024
#define COLUMNS_IN_ONCE 320
#define AMBIENT_COLOR make_float3(1,1,1)
#define EPSILON 0.01f
#define INF 1<<24
#define BACKGROUND make_float3(0,0,0)
#define WARP_SIZE 32


//int cudaMain(int argc, char **argv);
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#ifdef __JETBRAINS_IDE__
#include "math.h"
#define __CUDACC__ 1
#define __host__
#define __device__
#define __global__
#define __noinline__
#define __forceinline__
#define __shared__
#define __constant__
#define __managed__
#define __restrict__
// CUDA Synchronization
inline void __syncthreads() {};
inline void __threadfence_block() {};
inline void __threadfence() {};
inline void __threadfence_system();
inline int __syncthreads_count(int predicate) {return predicate;};
inline int __syncthreads_and(int predicate) {return predicate;};
inline int __syncthreads_or(int predicate) {return predicate;};
template<class T> inline T __clz(const T val) { return val; }
template<class T> inline T __shfl_down(const T val, int mask) { return val; }
template<class T> inline T __ldg(const T* address){return *address;};
// CUDA TYPES
typedef unsigned short uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned long long ulonglong;
typedef long long longlong;

//Atomic functions
int atomicMax(int* address, int val);
unsigned int atomicMax(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicMax(unsigned long long int* address,
                                 unsigned long long int val);

#endif


#endif //CUDA_BASE_CUDAHEADERS_H
