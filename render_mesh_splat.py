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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
# from renderer.gaussian_renderer import render
from renderer.mesh_splat_renderer import render # [YC] change to mesh_splat_renderer
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from games import gaussianModelRender

from pytorch3d.io import load_objs_as_meshes

def render_set(gs_type, model_path, name, iteration, views, gaussians, pipeline, background,
                # >>>> [YC] add
                texture_obj_path : str = None,
                occlusion: bool = False,
                policy_path : str = None
                # <<<< [YC] add
                ):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"renders_{gs_type}")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    textured_mesh = load_objs_as_meshes([texture_obj_path]).to("cuda") # [YC] add
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # rendering = render(view, gaussians, pipeline, background)["render"]
        if occlusion:
            rendering = render(view, gaussians, pipeline, 
                            bg_color=None, bg_depth=None,
                            textured_mesh=textured_mesh)["render"] # [YC] using different rasterizer
        else: # [YC] use original diff-gaussian-rasterizer for training
            pure_bg_depth = torch.full((1, view.image_height, view.image_width), 0, dtype=torch.float32, device="cuda")
            rendering = render(view, gaussians, pipeline, 
                            bg_color=None, bg_depth=pure_bg_depth,
                            textured_mesh=textured_mesh)["render"] # [YC] no occlusion handling, always use pure bg and pure depth
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(gs_type: str, dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                # >>>> [YC] add
                texture_obj_path : str = None,
                occlusion: bool = False,
                policy_path : str = None
                # <<<< [YC] add
                ):
    with torch.no_grad():
        gaussians = gaussianModelRender[gs_type](dataset.sh_degree)
        scene = Scene(dataset, gaussians, 
                      load_iteration=iteration, shuffle=False,
                      policy_path=policy_path)
        if hasattr(gaussians, 'update_alpha'):
            gaussians.update_alpha()
        if hasattr(gaussians, 'prepare_vertices'):
            gaussians.prepare_vertices()
        if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(gs_type, dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,
                        # >>>> [YC] add
                        texture_obj_path=texture_obj_path,
                        occlusion=occlusion,
                        policy_path=policy_path
                        # <<<< [YC] add
                        )

        if not skip_test:
             render_set(gs_type, dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,
                        # >>>> [YC] add
                        texture_obj_path=texture_obj_path,
                        occlusion=occlusion,
                        policy_path=policy_path
                        # <<<< [YC] add
                        )

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--gs_type', type=str, default="gs_flat")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    # >>>> [YC] add
    parser.add_argument("--texture_obj_path", type=str, default=None, help="Path to the textured obj file for mesh-based datasets.")
    parser.add_argument("--occlusion", action="store_true", help="Whether to use occlusion handling during rendering.")
    parser.add_argument("--policy_path", type=str, default="", help="Path to the splat density policy npy file.")
    # <<<< [YC] add
    parser.add_argument("--total_splats", type=int, default=131_072, help="Total number of splats to allocate (default: 2^17=131072)")
    parser.add_argument("--alloc_policy", type=str, default="area", help="Allocation policy for splats (default: area)")
    
    
    
    args = get_combined_args(parser) # get args from both command line and stored file
    model.gs_type = args.gs_type
    model.num_splats = args.num_splats
    
    # >>>> [SAM] add
    model.total_splats = args.total_splats
    model.alloc_policy = args.alloc_policy
    # <<<< [SAM] add
    
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(args.gs_type, model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                # >>>> [YC] add
                texture_obj_path=args.texture_obj_path,
                occlusion=args.occlusion,
                policy_path=args.policy_path
                # <<<< [YC] add
                )