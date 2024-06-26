from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def polygon_mask(gt_mask, num_sector, debug):
    return _RadPolygonMask.apply(gt_mask, num_sector, debug)


class _RadPolygonMask(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        gt_mask,
        num_sector, 
        debug=False,
    ):
        B, C, H, W = gt_mask.shape
        assert gt_mask.size(1) == 1, "shape at axis 1 shall be 1"
        args = (
            gt_mask,
            num_sector,
            B, H, W,
            debug,
        )

        num_sector, intersection_points, ray_angles, ray_dists = _C.polygon_mask(*args)
        return intersection_points, ray_angles, ray_dists


    @staticmethod
    def backward(ctx, grad_intersection, grad_rayangles):
        raise NotImplementedError