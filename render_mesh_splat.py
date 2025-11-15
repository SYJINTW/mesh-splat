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
from pathlib import Path
from PIL import Image
import torchvision
import torchvision.transforms as T
# from renderer.gaussian_renderer import render
from renderer.mesh_splat_renderer import render # [YC] change to mesh_splat_renderer
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from games import gaussianModelRender

from pytorch3d.io import load_objs_as_meshes
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from train import load_textured_mesh


def render_set(gs_type, model_path, name, iteration, views, gaussians, pipeline, background,
                # >>>> [YC] add
                texture_obj_path : str = None,
                occlusion: bool = False,
                policy_path : str = None,
                mesh_type : str = "colmap",
                textured_mesh = None,
                precaptured_mesh_img_path : str = None
                # <<<< [YC] add
                ):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"renders_{gs_type}")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # Load precaptured mesh background and depth if available
        bg = None
        bg_depth = None
        
        if precaptured_mesh_img_path:
            cached_bg_path = Path(precaptured_mesh_img_path) / "test_mesh_texture" / f"{view.image_name}.png"
            cached_bg_depth_path = Path(precaptured_mesh_img_path) / "test_mesh_depth" / f"{view.image_name}.pt"
            
            if cached_bg_path.exists():
                img = Image.open(cached_bg_path).convert("RGB")
                img = img.resize((view.image_width, view.image_height), Image.BILINEAR)
                transform = T.Compose([T.ToTensor()])
                bg = transform(img).to(torch.float32).cuda() # [0, 255] → [0.0, 1.0], shape (3, H, W)
                
            if cached_bg_depth_path.exists():
                bg_depth = torch.load(cached_bg_depth_path).unsqueeze(0).to("cuda")
        
        # [DONE] add pure GS renderer back here
        if gs_type == "gs":
            # Pure GS rendering without textured mesh
            pure_bg_template = background
            pure_bg = torch.tensor(pure_bg_template, dtype=torch.float32, device="cuda").view(3, 1, 1)
            pure_bg = pure_bg.expand(3, view.image_height, view.image_width)
            pure_bg_depth = torch.full((1, view.image_height, view.image_width), 0, dtype=torch.float32, device="cuda")
            
            rendering = render(view, gaussians, pipeline, 
                            bg_color=pure_bg, bg_depth=pure_bg_depth)["render"]
            print("\033[94m [INFO] Render::GS using pure GS rasterizer\033[0m")
            
        elif gs_type == "gs_mesh":
            # [NOTE] ensure that during rendering we use the same rasterizer as in training
            if occlusion:
                rendering = render(view, gaussians, pipeline, 
                                bg_color=bg, bg_depth=bg_depth,
                                textured_mesh=textured_mesh,
                                mesh_background_color=background)["render"] # [YC] using different rasterizer
                print("\033[92m [INFO] Render::DTGS using Depth+Texture+GS rasterizer for gs_mesh\033[0m")
                
            else: 
                pure_bg_depth = torch.full((1, view.image_height, view.image_width), 0, dtype=torch.float32, device="cuda")
                rendering = render(view, gaussians, pipeline, 
                                bg_color=bg, bg_depth=pure_bg_depth,
                                textured_mesh=textured_mesh,
                                mesh_background_color=background)["render"] # [YC] no occlusion handling, always use pure bg and pure depth
                print("\033[96m [INFO] Render::TGS using Texture+GS rasterizer for gs_mesh\033[0m")
                
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

# sets are {train,test, (val)}
def render_sets(gs_type: str, dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                # >>>> [YC] add
                texture_obj_path : str = None,
                occlusion: bool = False,
                policy_path : str = None,
                precaptured_mesh_img_path : str = None
                # <<<< [YC] add
                ):
    with torch.no_grad():
        gaussians = gaussianModelRender[gs_type](dataset.sh_degree)
        textured_mesh = load_textured_mesh(dataset=dataset, texture_obj_path=texture_obj_path)
        
        # [BUG] trace from here to see how ply and policy are loaded
        scene = Scene(dataset, gaussians, 
                      load_iteration=iteration, shuffle=False,
                      policy_path=policy_path,
                      texture_obj_path=texture_obj_path,
                      textured_mesh=textured_mesh
                      )
        if hasattr(gaussians, 'update_alpha'):
            gaussians.update_alpha()
        if hasattr(gaussians, 'prepare_vertices'):
            gaussians.prepare_vertices()
        if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()

        mesh_type = dataset.mesh_type if hasattr(dataset, 'mesh_type') else "sugar"
        print(f"[INFO] Render:: Using mesh type: {mesh_type}")
        
        # if mesh_type == "colmap":
        #     bg_color = [0,0,0] 
        #     print(f"[WARNING] Render:: overriding background color to black for colmap mesh type!")
        # else:
        #     print(f"[INFO] Render:: bg:{bg_color} for mesh type: {mesh_type}")
        
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        # [TODO] store more info about the scene into a json ("model card", "scene card")
        # [TODO] test dropping splats during rendering too
        # Add new params here
        render_kwargs = {
            'mesh_type': mesh_type,
            'texture_obj_path': texture_obj_path,
            'occlusion': occlusion,
            'policy_path': policy_path,
            'textured_mesh': scene.textured_mesh,
            'precaptured_mesh_img_path': precaptured_mesh_img_path
        }

        if not skip_train:
            render_set(gs_type, dataset.model_path, "train", scene.loaded_iter, 
                  scene.getTrainCameras(), gaussians, pipeline, background,
                  **render_kwargs)

        if not skip_test:
            render_set(gs_type, dataset.model_path, "test", scene.loaded_iter, 
                  scene.getTestCameras(), gaussians, pipeline, background,
                  **render_kwargs)

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
    parser.add_argument("--precaptured_mesh_img_path", type=str, default="",
        help="path to the directory containing precaptured mesh (RGB & D) images for background. \
            should contain mesh_texture/ and mesh_depth/ sub-folders.")
    # <<<< [YC] add
    
    # >>>> [Sam] add
    parser.add_argument("--total_splats", type=int, help="Total number of splats to allocate")
    parser.add_argument("--alloc_policy", type=str, default="area", help="Allocation policy for splats (default: area)")
    parser.add_argument("--budget_per_tri", type=float, default=1.0, help="set the total number of splats to be this number * number of triangles")
    # parser.add_argument("--drop_budget", type=int, help="drop until only this number of splats remain in the scene.")
    parser.add_argument('--mesh_type', type=str, default="sugar", help="textured mesh type: sugar, colmap, or others")
    
    # <<<< [Sam] add
    
    
    args = get_combined_args(parser) # get args from both command line and stored file
    model.gs_type = args.gs_type
    model.num_splats = args.num_splats
    
    # >>>> [SAM] add
    model.total_splats = args.total_splats
    model.alloc_policy = args.alloc_policy
    model.budget_per_tri = args.budget_per_tri
    model.mesh_type = args.mesh_type
    # model.drop_budget = args.drop_budget
    # <<<< [SAM] add
    
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(args.gs_type, model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                # >>>> [YC] add
                texture_obj_path=args.texture_obj_path,
                occlusion=args.occlusion,
                policy_path=args.policy_path,
                precaptured_mesh_img_path=args.precaptured_mesh_img_path
                # <<<< [YC] add
                )