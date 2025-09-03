import os
import argparse
import torch
import trimesh
import numpy as np
# from chamferdist import ChamferDistance
from pytorch3d.loss import chamfer_distance
from kaolin.metrics.pointcloud import chamfer_distance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, default="")
    parser.add_argument("--pred_path", type=str, default="")
    args = parser.parse_args()

    gt_path = args.gt_path
    pred_path = args.pred_path
    
    gt_points = np.asarray(trimesh.load(gt_path).vertices)
    gt_points = torch.from_numpy(gt_points[None]).float().cuda()
    print(gt_points.shape)
    pred_points = np.asarray(trimesh.load(pred_path).vertices)
    pred_points = torch.from_numpy(pred_points[None]).float().cuda()
    print(pred_points.shape)
    
    # chamfer_distance, _ = chamfer_distance(pred_points, gt_points)
    # chamfer_distance = chamfer_distance * 1000.
    
    dist = chamfer_distance(pred_points, gt_points, squared=False)
    print(dist * 100.0)