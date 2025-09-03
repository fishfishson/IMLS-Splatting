import os
import argparse
import numpy as np
import trimesh
from trimesh.remesh import subdivide_loop
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
    
    mesh = trimesh.load(args.save_path)
    print(f"Mesh vertices shape: {mesh.vertices.shape}")
    if mesh.vertices.shape[0] < 100000:
        print("Subdividing mesh")
        nv, nf = subdivide_loop(mesh.vertices, mesh.faces)
        mesh = trimesh.Trimesh(vertices=nv, faces=nf)
        mesh.export(os.path.join(os.path.dirname(args.save_path), "pred-high.ply"))
    else:
        mesh.export(os.path.join(os.path.dirname(args.save_path), "pred-high.ply"))