import os, copy, sys, uuid, warnings, json, re
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import random

from tqdm import tqdm
from fused_ssim import fused_ssim
from rasterizer import imlsplatting
from scene import Scene, GaussianModel
from utils.loss_utils import l1_loss
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import inverse_sigmoid
from refer import rasterize_set, eval_cd_dtu, eval_psnr_nerf, eval_cd_nerf

torch.set_num_threads(10)
warnings.filterwarnings("ignore")

def prepare_output_and_logger(args, expname):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
            unique_str = unique_str[0:10]
        else:
            unique_str = str(uuid.uuid4())
            unique_str = unique_str[0:10]
        if expname != '':
            unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

def training(dataset, opt, saving_iterations, checkpoint_ply, expname):
    prepare_output_and_logger(dataset, expname)
    gaussians = GaussianModel(opt)
    scene     = Scene(dataset, gaussians, checkpoint_ply=checkpoint_ply)
    gaussians.training_setup(opt)

    opt.model_path = scene.model_path
    with open(scene.model_path + '/opt.json', 'w') as json_file:
        json.dump(opt.__dict__, json_file, indent=4)
    imlsplat = imlsplatting(opt.__dict__)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end   = torch.cuda.Event(enable_timing = True)
    bg         = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    first_iter       = 0
    viewpoint_stack  = None
    ema_loss_for_log = 0.0
    progress_bar     = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter      += 1

    with torch.no_grad():
        meshdict = imlsplat(gaussians, scene.getTrainCameras()[0], bg, record='record/'+str(0), cameras=scene.getTrainCameras())
    
    for iteration in range(first_iter, opt.iterations + 1):  
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1))

        train_bgcolor = [1, 1, 1] if random.randint(0, 1)==1 else [0, 0, 0]
        train_bg      = torch.tensor(train_bgcolor, dtype=torch.float32, device="cuda")

        meshdict      = imlsplat(gaussians, viewpoint_cam, train_bg)

        gt_image  = viewpoint_cam.original_image.cuda().permute(1,2,0)
        if viewpoint_cam.gt_alpha_mask is not None:
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda().permute(1,2,0)
            gt_image = gt_image * gt_alpha_mask + train_bg * (1 - gt_alpha_mask)

        loss_L1   = l1_loss(meshdict['image'][..., 0:3].permute(2,0,1), gt_image.permute(2,0,1))
        loss_ssim = fused_ssim(meshdict['image'][..., 0:3].permute(2,0,1).unsqueeze(0), gt_image.permute(2,0,1).unsqueeze(0))
        loss      = (1.0 - opt.lambda_image) * loss_L1 + opt.lambda_image * (1.0 - loss_ssim)
            
        loss.backward()

        iter_end.record()
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # upsample
            if (iteration <= opt.upsample_until) and (iteration > opt.upsample_from) and ((iteration - opt.upsample_from) % opt.upsample_interval == 0):
                opt.grid_resolution = opt.grid_resolution + 64
                imlsplat = imlsplatting(opt.__dict__)
                print('upsample to ', str(opt.grid_resolution))

                gaussians.scale_range = True
                max_scale  = (opt.meshscale / (opt.grid_resolution - 1) * 1.5)
                init_scale = (opt.meshscale / (opt.grid_resolution - 1) * 1.0)
                gaussians.scale_max_val = torch.ones([gaussians.get_xyz.shape[0],1]).cuda() * max_scale

                new_scaling = inverse_sigmoid(init_scale / gaussians.scale_max_val)
                new_xyz     = gaussians._xyz
                new_normal  = gaussians._normal
                new_feature = gaussians._feature
                
                gaussians.prune_points(torch.ones(gaussians.get_xyz.shape[0]).bool().cuda())
                gaussians.densify_points({"xyz": new_xyz, "normal": new_normal, "scaling": new_scaling, "feature": new_feature})
                upsample = True
            else:
                upsample = False

            # resample
            if (not upsample) and (iteration <= opt.resample_until) and (iteration > opt.resample_from) and (iteration % opt.resample_interval == 0):
                opt_mesh = imlsplat(gaussians, scene.getTrainCameras()[0], bg, resample=True, cameras=scene.getTrainCameras())
                
            # record
            if (iteration % 500 == 0):
                meshdict = imlsplat(gaussians, scene.getTrainCameras()[0], bg, record='record/'+str(iteration), cameras=scene.getTrainCameras())

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint_ply", type=str, default = None)
    parser.add_argument("--expname", type=str, default='')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    # Start GUI server, configure and run training
    training(lp.extract(args), op.extract(args), args.save_iterations, args.start_checkpoint_ply, args.expname)

    # All done
    print("\nTraining complete.")
