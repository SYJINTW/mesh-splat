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
from renderer.mesh_splat_renderer import render_animated
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from games import gaussianModelRender

from pytorch3d.io import load_objs_as_meshes
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from train import load_textured_mesh

import json
import time


def transform_ficus_sinus(vertices, t, idxs):
    vertices[idxs, 2] += 0.005 * torch.sin(vertices[idxs, 0] * 2 * torch.pi + t)  # sinus
    vertices[idxs, 2] += 0.005 * torch.sin(vertices[idxs, 1] * 5 * torch.pi + t)  # sinus
    return vertices


def transform_hotdog_fly(vertices, t, idxs):
    vertices_new = vertices.clone()
    f = torch.sin(t) * 0.5
    vertices_new[:, 2] += t * (vertices[:, 0] ** 2 + vertices[:, 1] ** 2) ** (1 / 2) * 0.01
    return vertices_new


def transform_ficus_pot(vertices, t, idxs):
    if t > 8 * torch.pi:
        vertices[idxs, 2] += 0.005 * torch.sin(vertices[idxs, 1] * 5 * torch.pi + t)
    else:
        vertices[idxs, 2] -= (0.005 + t) * (vertices[idxs, 0] / 10) ** 2
    return vertices


def transform_ship_sinus(vertices, t, idxs=None):
    f = torch.sin(t) * 0.5
    vertices[:, 2] += 0.05 * torch.sin(vertices[:, 0] * torch.pi + f)  # sinus
    return vertices


def make_smaller(vertices, t, idxs=None):
    vertices_new = vertices.clone()
    f = torch.sin(t) + 1
    vertices_new = f * vertices_new
    return vertices_new


def do_not_transform(vertices, t):
    return vertices


def render_set(gs_type, model_path, name, iteration, views, gaussians, pipeline, background,
               texture_obj_path: str = None,
               occlusion: bool = False,
               policy_path: str = None,
               mesh_type: str = "colmap",
               textured_mesh=None,
               precaptured_mesh_img_path: str = None,
               transform_func=None):
    """
    Render animated sequence with mesh deformation.
    """
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"renders_animated_{gs_type}")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_animated")
    debug_path = os.path.join(model_path, name, "ours_{}".format(iteration), "debug_animated")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(debug_path, exist_ok=True)

    # Time sequence for animation
    t = torch.linspace(0, 10 * torch.pi, len(views))
    
    # Get initial vertices
    vertices = gaussians.vertices.clone()
    
    # Choose indexes if you want to change only part of the mesh
    idxs = None

    for idx, view in enumerate(tqdm(views, desc="Rendering animated progress")):
        # Debug flag for saving intermediate images
        debug_flag = (idx % 10 == 0)
        
        # Apply transformation to vertices
        if transform_func is not None:
            new_vertices = transform_func(vertices.clone(), t[idx], idxs)
        else:
            new_vertices = vertices
        
        # Create triangles from transformed vertices
        triangles = new_vertices[torch.tensor(gaussians.faces).long()].float().cuda()
        
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
                bg = transform(img).to(torch.float32).cuda()
                if debug_flag:
                    torchvision.utils.save_image(bg, os.path.join(debug_path, '{0:05d}_bg'.format(idx) + ".png"))
            
            if cached_bg_depth_path.exists():
                bg_depth = torch.load(cached_bg_depth_path).unsqueeze(0).to("cuda")
        
        # Render based on gs_type
        if gs_type == "gs":
            # Pure GS rendering without textured mesh
            pure_bg_template = background
            pure_bg = torch.tensor(pure_bg_template, dtype=torch.float32, device="cuda").view(3, 1, 1)
            pure_bg = pure_bg.expand(3, view.image_height, view.image_width)
            pure_bg_depth = torch.full((1, view.image_height, view.image_width), 0, dtype=torch.float32, device="cuda")
            
            rendering = render_animated(idxs, triangles, view, gaussians, pipeline,
                                       bg_color=pure_bg, bg_depth=pure_bg_depth)["render"]
            print("\033[94m [INFO] AnimatedRender::GS using pure GS rasterizer\033[0m")
            
        elif gs_type == "gs_mesh":
            if occlusion:
                rendering = render_animated(idxs, triangles, view, gaussians, pipeline,
                                           bg_color=bg, bg_depth=bg_depth,
                                           textured_mesh=textured_mesh,
                                           mesh_background_color=background)["render"]
                print("\033[92m [INFO] AnimatedRender::DTGS using Depth+Texture+GS rasterizer\033[0m")
            else:
                pure_bg_depth = torch.full((1, view.image_height, view.image_width), 0, dtype=torch.float32, device="cuda")
                rendering = render_animated(idxs, triangles, view, gaussians, pipeline,
                                           bg_color=bg, bg_depth=pure_bg_depth,
                                           textured_mesh=textured_mesh,
                                           mesh_background_color=background)["render"]
                print("\033[96m [INFO] AnimatedRender::TGS using Texture+GS rasterizer\033[0m")
            
            # Debug: save pure gaussian
            if debug_flag:
                _pure_bg_template = background
                _pure_bg = torch.tensor(_pure_bg_template, dtype=torch.float32, device="cuda").view(3, 1, 1)
                _pure_bg = _pure_bg.expand(3, view.image_height, view.image_width)
                _pure_bg_depth = torch.full((1, view.image_height, view.image_width), 0, dtype=torch.float32, device="cuda")
                _rendering = render_animated(idxs, triangles, view, gaussians, pipeline,
                                            bg_color=_pure_bg, bg_depth=_pure_bg_depth)["render"]
                torchvision.utils.save_image(_rendering, os.path.join(debug_path, '{0:05d}_pure_gs'.format(idx) + ".png"))
        
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        # Save debug images
        if debug_flag:
            torchvision.utils.save_image(rendering, os.path.join(debug_path, '{0:05d}_rendering'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(debug_path, '{0:05d}_gt'.format(idx) + ".png"))


def render_sets(gs_type: str, dataset: ModelParams, iteration: int, pipeline: PipelineParams, 
                skip_train: bool, skip_test: bool,
                texture_obj_path: str = None,
                occlusion: bool = False,
                policy_path: str = None,
                precaptured_mesh_img_path: str = None,
                transform_name: str = "hotdog_fly"):
    """
    Render animated sets for train/test views.
    """
    render_timer_start = time.time()
    
    # Select transformation function
    transform_funcs = {
        "ficus_sinus": transform_ficus_sinus,
        "hotdog_fly": transform_hotdog_fly,
        "ficus_pot": transform_ficus_pot,
        "ship_sinus": transform_ship_sinus,
        "make_smaller": make_smaller,
        "none": do_not_transform
    }
    transform_func = transform_funcs.get(transform_name, transform_hotdog_fly)
    print(f"[INFO] Using transformation: {transform_name}")
    
    with torch.no_grad():
        gaussians = gaussianModelRender[gs_type](dataset.sh_degree)
        textured_mesh = load_textured_mesh(dataset=dataset, texture_obj_path=texture_obj_path)
        
        scene = Scene(dataset, gaussians,
                     load_iteration=iteration, shuffle=False,
                     policy_path=policy_path,
                     texture_obj_path=texture_obj_path,
                     textured_mesh=textured_mesh)
        
        if hasattr(gaussians, 'update_alpha'):
            gaussians.update_alpha()
        if hasattr(gaussians, 'prepare_vertices'):
            gaussians.prepare_vertices()
        if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()

        mesh_type = dataset.mesh_type if hasattr(dataset, 'mesh_type') else "sugar"
        print(f"[INFO] AnimatedRender:: Using mesh type: {mesh_type}")
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_kwargs = {
            'mesh_type': mesh_type,
            'texture_obj_path': texture_obj_path,
            'occlusion': occlusion,
            'policy_path': policy_path,
            'textured_mesh': scene.textured_mesh,
            'precaptured_mesh_img_path': precaptured_mesh_img_path,
            'transform_func': transform_func
        }

        if not skip_train:
            render_set(gs_type, dataset.model_path, "train", scene.loaded_iter,
                      scene.getTrainCameras(), gaussians, pipeline, background,
                      **render_kwargs)

        if not skip_test:
            render_set(gs_type, dataset.model_path, "test", scene.loaded_iter,
                      scene.getTestCameras(), gaussians, pipeline, background,
                      **render_kwargs)
    
    render_time = time.time() - render_timer_start
    print(f"[INFO] Total animated rendering time: {render_time:.2f} seconds")


if __name__ == "__main__":
    parser = ArgumentParser(description="Animated rendering script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--gs_type', type=str, default="gs_mesh")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    # Mesh and rendering options
    parser.add_argument("--texture_obj_path", type=str, default=None)
    parser.add_argument("--occlusion", action="store_true")
    parser.add_argument("--policy_path", type=str, default="")
    parser.add_argument("--precaptured_mesh_img_path", type=str, default="")
    
    # Animation options
    parser.add_argument("--transform", type=str, default="hotdog_fly",
                       choices=["ficus_sinus", "hotdog_fly", "ficus_pot", "ship_sinus", "make_smaller", "none"],
                       help="Transformation to apply to mesh vertices")
    
    # Budget and allocation
    parser.add_argument("--total_splats", type=int)
    parser.add_argument("--alloc_policy", type=str, default="area")
    parser.add_argument("--budget_per_tri", type=float, default=1.0)
    parser.add_argument('--mesh_type', type=str, default="sugar")
    
    args = get_combined_args(parser)
    model.gs_type = args.gs_type
    model.num_splats = args.num_splats
    model.total_splats = args.total_splats
    model.alloc_policy = args.alloc_policy
    model.budget_per_tri = args.budget_per_tri
    model.mesh_type = args.mesh_type
    
    print("Rendering animated " + args.model_path)

    safe_state(args.quiet)

    render_sets(args.gs_type, model.extract(args), args.iteration, pipeline.extract(args),
               args.skip_train, args.skip_test,
               texture_obj_path=args.texture_obj_path,
               occlusion=args.occlusion,
               policy_path=args.policy_path,
               precaptured_mesh_img_path=args.precaptured_mesh_img_path,
               transform_name=args.transform)