#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>


std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor>
PolygonMask(
	const torch::Tensor& gt_mask, // shape (1, h, w)
    const int num_sector,
    const int B,
    const int H,
    const int W,
	const bool debug);