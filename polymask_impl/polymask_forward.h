#ifndef POLYMASK_IMPL_H_INCLUDED
#define POLYMASK_IMPL_H_INCLUDED

#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <functional>
#include "stdio.h"


#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

#define MAX_ITER 10


namespace POLY_MASK_FORWARD {
    void ray_intersection(const float* gt_mask,
                            const int sector_num,
                            const int B,
                            const int H, 
                            const int W,
                            float* intercection_points,
                            float* ray_angles,
                            float* ray_dists);
};

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif