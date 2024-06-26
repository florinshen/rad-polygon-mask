#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>

#include "polymask_impl/polymask_forward.h"
#include <fstream>
#include <string>
#include <functional>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor>
PolygonMask(
	const torch::Tensor& gt_mask, // shape (1, h, w)
    const int num_sector,
	const int B,
    const int H,
    const int W,
	const bool debug)
{

	auto int_opts = gt_mask.options().dtype(torch::kInt32);
  	auto float_opts = gt_mask.options().dtype(torch::kFloat32);

	// printf("enter PolygonMask, with sector_num:  %d\n", num_sector);

	torch::Tensor intersection_points = torch::full({B, num_sector, 2}, -1.0, float_opts);
	torch::Tensor ray_angles = torch::full({B, num_sector}, -1.0, float_opts);
	torch::Tensor ray_dists = torch::full({B, num_sector}, -1.0, float_opts);

	POLY_MASK_FORWARD::ray_intersection(
		gt_mask.contiguous().data_ptr<float>(),
		num_sector,
		B,
		H, 
		W,
		intersection_points.contiguous().data_ptr<float>(),
		ray_angles.contiguous().data_ptr<float>(),
		ray_dists.contiguous().data_ptr<float>()
	);
	return std::make_tuple(num_sector, intersection_points, ray_angles, ray_dists);
}