import os
import argparse
import torch
import trimesh
import numpy as np
# from chamferdist import ChamferDistance
# from pytorch3d.loss import chamfer_distance
from kaolin.metrics.pointcloud import chamfer_distance, sided_distance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, default="")
    parser.add_argument("--pred_path", type=str, default="")
    parser.add_argument("--mode", type=str, default="body", choices=["body", "head"])
    args = parser.parse_args()

    gt_path = args.gt_path
    pred_path = args.pred_path
    
    gt_points = np.asarray(trimesh.load(gt_path).vertices)
    gt_points = torch.from_numpy(gt_points[None]).float().cuda()
    pred_points = np.asarray(trimesh.load(pred_path).vertices)
    pred_points = torch.from_numpy(pred_points[None]).float().cuda()
    
    if args.mode == "body":
        chamfer_dist = chamfer_distance(gt_points, pred_points, squared=False)
        print("error in chamfer distance : ", chamfer_dist * 100.0)
    elif args.mode == "head":
        sided_dist, _ = sided_distance(gt_points, pred_points)
        sided_dist = torch.sqrt(sided_dist).mean()
        print("error in sided distance : ", sided_dist * 100.0)
    else:
        print("Invalid mode")
        exit(1)