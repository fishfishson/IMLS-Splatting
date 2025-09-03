import os
import argparse
import numpy as np
import trimesh
import json
import torch
from scene import Scene, GaussianModel
from rasterizer import imlsplatting


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--save_path", type=str, default="debug/test.ply")
    parser.add_argument("--grid_resolution", type=int, default=576)
    args = parser.parse_args()

    config = args.config
    model_path = os.path.join(args.model_path, "point_cloud/iteration_20000/point_cloud.ply")


    with open(config, "r") as f:
        config = json.load(f)

    gaussians = GaussianModel(config)
    gaussians.create_from_ply(model_path, 1.0)
    
    imlsplat = imlsplatting(config)
    
    with torch.no_grad():
        meshdict = imlsplat.extract(gaussians, os.path.join(args.save_path), args.grid_resolution)