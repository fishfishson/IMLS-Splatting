#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
import cv2
import torch
from utils.graphics_utils import fov2focal
from scene.dataset_readers import CameraInfoEasy, CameraInfo

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_h, orig_w = cam_info.image.shape[:2]

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        scale = resolution_scale * args.resolution
    else:  # should be a type that converts to float
        raise NotImplementedError("Resolution scale not implemented for this resolution")
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    # resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    resized_image_rgb = cv2.resize(cam_info.image, resolution, interpolation=cv2.INTER_AREA)
    resized_image_rgb = resized_image_rgb.astype(np.float32) / 255.0
    resized_image_rgb = torch.from_numpy(resized_image_rgb)
    if resized_image_rgb.ndim == 3:
        resized_image_rgb = resized_image_rgb.permute(2, 0, 1)
    else:
        resized_image_rgb = resized_image_rgb.unsqueeze(dim=-1).permute(2, 0, 1)
    if hasattr(cam_info, 'mask'):
        # resized_image_mask = PILtoTorch(cam_info.mask, resolution)
        resized_image_mask = cv2.resize(cam_info.mask, resolution, interpolation=cv2.INTER_AREA)
        resized_image_mask = resized_image_mask.astype(np.float32) / 255.0
        resized_image_mask = torch.from_numpy(resized_image_mask)
        if resized_image_mask.ndim == 3:
            resized_image_mask = resized_image_mask.permute(2, 0, 1)
        else:
            resized_image_mask = resized_image_mask.unsqueeze(dim=-1).permute(2, 0, 1)
    else:
        resized_image_mask = None

    gt_image = resized_image_rgb[:3, ...]
    if resized_image_mask is not None:
        loaded_mask = resized_image_mask[:1, ...]
    else:
        loaded_mask = None
    
    if isinstance(cam_info, CameraInfoEasy):
        cx = cam_info.K[0, 2] / scale
        cy = cam_info.K[1, 2] / scale
        fl_x = cam_info.K[0, 0] / scale
        fl_y = cam_info.K[1, 1] / scale
        return Camera(
            colmap_id=cam_info.name, R=cam_info.R, T=cam_info.T,
            FoVx=-1.0, FoVy=-1.0,
            image=gt_image, gt_alpha_mask=loaded_mask,
            image_name=cam_info.image_name, uid=id, data_device=args.data_device,
            cx=cx, cy=cy, fl_x=fl_x, fl_y=fl_y
        )
    elif isinstance(cam_info, CameraInfo):
        return Camera(
            colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
            FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
            image=gt_image, gt_alpha_mask=loaded_mask,
            image_name=cam_info.image_name, uid=id, data_device=args.data_device
        )

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
