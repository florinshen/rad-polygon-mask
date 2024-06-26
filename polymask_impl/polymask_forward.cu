#include "polymask_forward.h"
#include <cmath>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


// cpu version
// bool isPointInside(const float* mask, 
//                     const int width, 
//                     const float2* point,
//                     bool debug) {
//     int y =  static_cast<int>(point->y);
//     int x =  static_cast<int>(point->x);
//     float mask_val = mask[y * width + x];
//     if (debug){
//         printf("check points at position (%d, %d), with value %.1f\n", x, y, mask_val);
//     }
//     return mask_val > 0;
// }


// cuda kernel version
__device__ void isPointInside(uint8_t* flag,
                    const float* mask, 
                    const int width, 
                    const float2* point,
                    bool debug) {
    int y =  static_cast<int>(point->y);
    int x =  static_cast<int>(point->x);
    float mask_val = mask[y * width + x];
    if (debug){
        printf("check points at position (%d, %d), with value %.1f\n", x, y, mask_val);
    }
    *flag = mask_val > 0 ? 1:0;
}




__device__ void getMid(
    float2* mid,
    const float2 *start, 
    const float2 *end){
    mid->x = (start->x + end->x) / 2;
    mid->y = (start->y + end->y) / 2;
}

__device__ void getDist(float* dist, const float2* p1, const float2* p2) {
    *dist = std::sqrt(std::pow(p2->x - p1->x, 2) + std::pow(p2->y - p1->y, 2));
}


__global__ void _cuda_ray_intersection(
    const float* gt_mask, 
    const int B, 
    const int H, 
    const int W,
    const int sector_num,
    const float angle_step,
    float*  intersection_points,
    float*  ray_angles,
    float* ray_dists
){
    uint32_t batch_idx = blockIdx.x;
    uint32_t ray_idx = threadIdx.x;
    float centerX = (W - 1) / 2.0;
    float centerY = (H - 1) / 2.0;

    bool debug=false;
    // if (ray_idx == 5){
    //     debug = true;
    // }

    float2 startPoint = {centerX, centerY};

    float angle = ray_idx * angle_step;
    float gradient = tan(angle);
    float2 endPoint;

    // Determine if the line intersects with the top/bottom or left/right borders
    if (debug){
        printf("angle is %.2f PI, with ray_idx %d\n", angle / M_PI, ray_idx);
    }

    if ((angle <= M_PI / 4 ) || (angle > 3 * M_PI / 4 && angle <= 5 * M_PI / 4) || (angle > 7 * M_PI / 4)) {
        // Intersects with left or right border
        float targetX = (angle <= M_PI / 2 || angle > 3 * M_PI / 2) ? (W - 1) : 0;
        float deltaX = targetX - centerX;
        float deltaY = deltaX * gradient;
        endPoint.x = targetX;
        endPoint.y = centerY - deltaY;
    } else {
        // Intersects with top or bottom border
        float targetY = (angle > M_PI / 4 && angle <= 3 * M_PI / 4) ? 0 : (H - 1);
        float deltaY = targetY - centerY;
        float deltaX = deltaY / gradient;
        endPoint.x = centerX - deltaX;
        endPoint.y = targetY;
    }

    // Correct endpoints that exceed image borders due to rounding errors
    endPoint.x = std::max(0.0f, std::min(endPoint.x, static_cast<float>(W)));
    endPoint.y = std::max(0.0f, std::min(endPoint.y, static_cast<float>(H)));


    // printf("ray %d of %d th image, with sector_num %d, angle %.2f \n", ray_idx, batch_idx, sector_num, angle);

    uint32_t angle_idx = batch_idx * sector_num + ray_idx;    

    ray_angles[angle_idx] = angle;

    // then find find the intersection point with bisection

    float2 midpoint;
     // simple assumption, start is isinside, and end is outside, 
    float2 start = startPoint;
    float2 end = endPoint;

    // gt_mask shape (bsz, 1, h, w)
    // float* cur_mask = &gt_mask[batch_idx * H * W];
    int iter_num = 0;
    uint32_t point_idx = batch_idx * sector_num * 2 + ray_idx * 2;

    // intersection_points shape (bsz, sector_num, 2)

    uint8_t inside_flag = 0;
    uint32_t mask_offset = batch_idx * H * W;
    const float* cur_mask = gt_mask + mask_offset;

    isPointInside(&inside_flag, cur_mask, W, &endPoint, debug);
    if (inside_flag > 0){
        intersection_points[point_idx] = endPoint.x;
        intersection_points[point_idx + 1] = endPoint.y;
        getDist(ray_dists + angle_idx, &endPoint, &startPoint);
    } else {
        while (true){
            iter_num ++;
            getMid(&midpoint, &start, &end);
            if ((std::abs(start.x - end.x) < 0.1 && std::abs(start.y - end.y) < 0.1) || (iter_num > MAX_ITER)){
                intersection_points[point_idx] = midpoint.x;
                intersection_points[point_idx + 1] = midpoint.y;
                // get the L2 distance between border and center
                getDist(ray_dists + angle_idx, &midpoint, &startPoint);
                break;
            }
            isPointInside(&inside_flag, cur_mask, W, &midpoint, debug);
            if (inside_flag > 0){
                start = midpoint;
            } else {
                end = midpoint;
            }
        }
    }

}


void POLY_MASK_FORWARD::ray_intersection(
    const float* gt_mask,
    const int sector_num,
    const int B,
    const int H, 
    const int W,
    float* intersection_points,
    float* ray_angles,
    float* ray_dists
){
    // printf("batch_size: %d, with sector_num: %d\n", B, sector_num);
    dim3 grid(B, 1);
    dim3 block(sector_num, 1);
    // float centerX = W / 2.0;
    // float centerY = H / 2.0;
    float angleStep = 2 * M_PI / sector_num;

    _cuda_ray_intersection<<<grid, block>>>(gt_mask, 
                                            B, H, W, 
                                            sector_num,
                                            angleStep, 
                                            intersection_points, 
                                            ray_angles,
                                            ray_dists);
    // CHECK_CUDA(, true);
}