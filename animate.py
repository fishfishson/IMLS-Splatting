import os
import trimesh
import argparse
import smplx
import numpy as np
import torch

from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes

from utils.sample_utils import sample_closest_points_on_surface


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, default="./output/0103")
    parser.add_argument("--smplx_path", type=str, default="./data/thuman2.1_smplx")
    parser.add_argument("--pose_path", type=str, default="./data/pose/CMU/10/10_05_poses.npz")
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()

    pose_file = np.load(args.pose_path)
    smpl_data = {
        'global_orient': pose_file['poses'][::args.interval, :3],
        'transl': pose_file['trans'][::args.interval],
        'body_pose': pose_file['poses'][::args.interval, 3: 22 * 3],
    }
    smpl_data['body_pose'][:, 13 * 3 + 2] -= 0.3
    smpl_data['body_pose'][:, 12 * 3 + 2] += 0.3
    smpl_data['body_pose'][:, 19 * 3: 20 * 3] = 0.
    smpl_data['body_pose'][:, 20 * 3: 21 * 3] = 0.
    smpl_data['body_pose'][:, 14 * 3] = 0.
    smpl_data = {k: torch.from_numpy(v).to(torch.float32) for k, v in smpl_data.items()}

    mesh_path = os.path.join(args.mesh_path, "pred.ply")
    mesh_name = os.path.basename(os.path.dirname(mesh_path))
    smplx_path = os.path.join(args.smplx_path, mesh_name)
    
    v, f = load_ply(mesh_path)
    mesh = Meshes(verts=[v], faces=[f])
    mesh = mesh.to(torch.device('cuda'))

    smplx_model = smplx.SMPLX(model_path='./data/smplx', gender='neutral', ext='npz', use_pca=False)
    params = np.load(os.path.join(smplx_path, 'smplx_param.pkl'), allow_pickle=True)
    with torch.no_grad():
        smpl_out = smplx_model.forward(
            transl=torch.tensor(params['transl'], dtype=torch.float32),
            global_orient=torch.tensor(params['global_orient'], dtype=torch.float32),
            body_pose=torch.tensor(params['body_pose'], dtype=torch.float32),
            betas=torch.tensor(params['betas'], dtype=torch.float32),
            left_hand_pose=torch.tensor(params['left_hand_pose'], dtype=torch.float32),
            right_hand_pose=torch.tensor(params['right_hand_pose'], dtype=torch.float32),
            expression=torch.tensor(params['expression'], dtype=torch.float32),
            jaw_pose=torch.tensor(params['jaw_pose'], dtype=torch.float32),
            leye_pose=torch.tensor(params['leye_pose'], dtype=torch.float32),
            reye_pose=torch.tensor(params['reye_pose'], dtype=torch.float32)
        )
    smpl_mesh = Meshes(verts=[smpl_out.vertices[0]], faces=[torch.from_numpy(smplx_model.faces).long()])
    smpl_mesh = smpl_mesh.to(torch.device('cuda'))

    values = smplx_model.lbs_weights
    values = values.to(torch.device('cuda'))
    sampled, _ = sample_closest_points_on_surface(mesh.verts_padded(), smpl_mesh.verts_padded(), smpl_mesh.faces_padded(), values)

    A = smpl_out.A.to(torch.device('cuda'))
    inv_A = torch.linalg.inv(A)
    T = torch.einsum('bnj,bjkl->bnkl', sampled, inv_A)

    homo = torch.cat([mesh.verts_packed(), torch.ones_like(mesh.verts_packed())[:, :1]], dim=1)
    cano_v = torch.einsum('nl,nkl->nk', homo, T[0])[..., :3]

    for i in range(len(smpl_data['transl'])):
        with torch.no_grad():
            smpl_out = smplx_model.forward(
                transl=torch.tensor(smpl_data['transl'][i:i+1], dtype=torch.float32),
                global_orient=torch.tensor(smpl_data['global_orient'][i:i+1], dtype=torch.float32),
                body_pose=torch.tensor(smpl_data['body_pose'][i:i+1], dtype=torch.float32),
                betas=torch.tensor(params['betas'], dtype=torch.float32),
                left_hand_pose=torch.tensor(params['left_hand_pose'], dtype=torch.float32),
                right_hand_pose=torch.tensor(params['right_hand_pose'], dtype=torch.float32),
                expression=torch.tensor(params['expression'], dtype=torch.float32),
                jaw_pose=torch.tensor(params['jaw_pose'], dtype=torch.float32),
                leye_pose=torch.tensor(params['leye_pose'], dtype=torch.float32),
                reye_pose=torch.tensor(params['reye_pose'], dtype=torch.float32)
            )
            A = smpl_out.A.to(torch.device('cuda'))
        T = torch.einsum('bnj,bjkl->bnkl', sampled, A)
        homo = torch.cat([cano_v, torch.ones_like(cano_v[:, :1])], dim=1)
        T_v = torch.einsum('nl,nkl->nk', homo, T[0])[..., :3]
        name = os.path.basename(args.pose_path).split('.')[0]
        _ = trimesh.Trimesh(vertices=T_v.cpu().numpy(), faces=mesh.faces_packed().cpu().numpy()).export(os.path.join(args.mesh_path, f"{name}_{i:04d}.ply"))
