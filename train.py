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

# from pytorch3d.io import load_objs_as_meshes

from itertools import count
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
# from renderer.gaussian_renderer import render, network_gui
from renderer.mesh_splat_renderer import render, network_gui
import sys
from scene import Scene
from games import (
    optimizationParamTypeCallbacks,
    gaussianModel
)

from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import numpy as np
from pathlib import Path

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    AmbientLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    )

import open3d as o3d
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)
from scene.cameras import convert_camera_from_gs_to_pytorch3d
from pytorch3d.renderer.blending import BlendParams
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

from mesh_renderer_pytorch3d import mesh_renderer_pytorch3d
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

SCENE_NAME = "hotdog"

def load_with_white_bg(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # keep alpha if present

    if img.shape[2] == 4:  # RGBA
        rgb = img[:, :, :3].astype(np.float32) / 255.0
        alpha = img[:, :, 3:].astype(np.float32) / 255.0  # shape [H,W,1]
        white_bg = np.ones_like(rgb)
        img_out = rgb * alpha + white_bg * (1 - alpha)
    else:
        img_out = img[:, :, :3].astype(np.float32) / 255.0

    # Convert BGR to RGB because OpenCV loads in BGR order
    img_out = img_out[:, :, ::-1]
    return img_out

# [DONE] migrate this policy
# [TODO] [DOING] extract the warmup stage out of the main training loop
def warmup(viewpoint_cameras, p3d_mesh,
               image_height=800, image_width=800, faces_per_pixel=1,
               device="cuda"):
    """
    The main idea is using the visual quality to decide how many Gaussians should we set on each triangle.
    First, we render the mesh from each viewpoint camera, and compare with the ground truth images.
    Then, we can get the distortion map (pixel-wise difference).
    Second, we prioritize the triangles that project to high distortion pixels, and assign more Gaussians to them.
    """
    #[TODO] also migrate the debugging logic
    debugging = True
    if debugging:
        heatmap_output_dir = Path(f"../distortion_heatmap")
        heatmap_output_dir.mkdir(parents=True, exist_ok=True)
        mesh_bg_output_dir = Path(f"../mesh_bg")
        mesh_bg_output_dir.mkdir(parents=True, exist_ok=True)
        filtered_obj_output_dir = Path(f"../filtered_obj")
        filtered_obj_output_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh with Trimesh
    tm_mesh = trimesh.load("/mnt/data1/syjintw/NEU/dataset/hotdog/mesh.obj", process=False)
    
    # Change Trimesh into PyTorch3D, and also handle texture from UV map
    verts = torch.tensor(tm_mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(tm_mesh.faces, dtype=torch.int64, device=device)  # keep order!
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb)
    tm2p3d_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    
    # Load mesh with PyTorch3D
    p3d_mesh = load_objs_as_meshes(["/mnt/data1/syjintw/NEU/dataset/hotdog/mesh.obj"]).to(device)

    num_faces = faces.shape[0]
    dist_map_all = np.zeros(num_faces, dtype=np.float32)
    for i, viewpoint_camera in enumerate(viewpoint_cameras):
        # Load ground truth image
        gt_image_path = f"/mnt/data1/syjintw/NEU/dataset/hotdog/mesh_texture/{viewpoint_camera.image_name}.png"
        gt_img = load_with_white_bg(gt_image_path) # (W, H)
        
        # Render p3d_mesh with PyTorch3D for comparing the visual quality
        p3d_mesh_color, p3d_mesh_depth, p3d_fragments = mesh_renderer_pytorch3d(viewpoint_camera, p3d_mesh,
                                                            image_height=image_height, image_width=image_width,
                                                            faces_per_pixel=faces_per_pixel,
                                                            device=device)
        
        # Convert p3d_mesh_color to NumPy for computing distortion
        p3d_mesh_color_np = (
            p3d_mesh_color[0, ..., :3]              # (H, W, 3)
            .detach().cpu().numpy()                 # → NumPy
        )
        p3d_mesh_color_np = np.clip(p3d_mesh_color_np, 0.0, 1.0).astype(np.float32)
        
        # # Save rendered mesh background and heatmap for debugging
        # if debugging:
        #     # Transfer p3d_mesh_color to PIL Image for debugging
        #     p3d_mesh_color_pil = TF.to_pil_image(p3d_mesh_color.cpu())
        #     p3d_mesh_color_pil.save(mesh_bg_output_dir/f"r_{i}.png")
            
        #     # Compute heatmap (L2 difference)
        #     diff = np.sqrt(np.sum((gt_img - p3d_mesh_color_np) ** 2, axis=2))  # (H, W)
        #     diff_normalized = diff / np.max(diff)  # normalize to [0,1]

        #     # Plot & save heatmap
        #     plt.figure(figsize=(8, 8))
        #     plt.imshow(diff_normalized, cmap='hot')
        #     plt.axis('off')
        #     plt.tight_layout()
        #     plt.savefig(heatmap_output_dir/f"r_{i}.png", dpi=300, bbox_inches='tight', pad_inches=0)
        #     plt.close()
        
        # Compute per-pixel absolute difference map
        dist_map = np.mean(np.abs(gt_img - p3d_mesh_color_np), axis=2)  # shape [H, W]. It will be in [0, 1]
        # print("dist_map:", dist_map.shape, dist_map.dtype, dist_map.min(), dist_map.max(), dist_map.mean())
        
        # Render tm2p3d_mesh with PyTorch3D
        tm2p3d_mesh_color, tm2p3d_mesh_depth, tm2p3d_fragments = mesh_renderer_pytorch3d(viewpoint_camera, tm2p3d_mesh,
                                                                                        image_height=image_height, image_width=image_width,
                                                                                        faces_per_pixel=faces_per_pixel,
                                                                                        device=device)
        
        # Face index per pixel
        face_idx_map = tm2p3d_fragments.pix_to_face[0, ..., 0].cpu().numpy()  # (H, W). face_idx_map = -1 means background
        # print("face_idx_map:", face_idx_map.shape, face_idx_map.dtype, face_idx_map.min(), face_idx_map.max(), face_idx_map.mean())
        
        # Flatten arrays
        face_idx_flat = face_idx_map.flatten()
        dist_flat = dist_map.flatten()
        
        # Filter out invalid faces (e.g., -1)
        valid_mask = face_idx_flat >= 0
        face_idx_flat = face_idx_flat[valid_mask]
        dist_flat = dist_flat[valid_mask]
        
        # Compute sum and counts per face
        sum_dist = np.bincount(face_idx_flat, weights=dist_flat, minlength=num_faces)
        count = np.bincount(face_idx_flat, minlength=num_faces)

        # Avoid divide-by-zero
        mean_dist = np.zeros(num_faces, dtype=np.float32)
        mask = count > 0
        mean_dist[mask] = sum_dist[mask] / count[mask]
        
        # Update overall distortion map
        dist_map_all += mean_dist
    
    if debugging:
        dist_norm = (dist_map_all - dist_map_all.min()) / (dist_map_all.ptp() + 1e-8)
        cmap = cm.get_cmap('jet')
        colors = cmap(dist_norm)[:, :3]  # (num_faces, 3), RGB in [0,1]
        
        # Compute per-vertex color by averaging colors of adjacent faces
        vertex_colors = np.zeros((len(tm_mesh.vertices), 3))
        for f_id, verts in enumerate(tm_mesh.faces):
            vertex_colors[verts] += colors[f_id]
            
        counts = np.bincount(tm_mesh.faces.flatten(), minlength=len(tm_mesh.vertices))
        vertex_colors /= np.maximum(counts[:, None], 1e-8)

        # Build Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(tm_mesh.vertices)
        pcd.colors = o3d.utility.Vector3dVector(vertex_colors)

        # Save as PLY (works perfectly in MeshLab)
        o3d.io.write_point_cloud("../distortion_heatmap.ply", pcd)
    
    # Decide triangle filter based on distortion map
    print("dist_map_all:", dist_map_all.shape, dist_map_all.dtype, dist_map_all.min(), dist_map_all.max(), dist_map_all.mean())
    
    # Normalize distortion map to int in [0,5] based on the distortion values
    # Normalize to [0, 1]
    dist_norm = (dist_map_all - dist_map_all.min()) / (dist_map_all.max() - dist_map_all.min() + 1e-8)
    
    # Scale to [0, 5] and convert to int
    dist_int = np.round(dist_norm * 5).astype(np.int32)
    print("dist_int:", dist_int.shape, dist_int.dtype, dist_int.min(), dist_int.max(), dist_int.mean())

    print("Saving number of Gaussians in each triangle...")
    np.save("/mnt/data1/syjintw/NEU/dataset/hotdog/num_of_gaussians.npy", dist_int)
   
def training(gs_type, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
            debug_from, save_xyz,
            # >>>> [YC] add
            texture_obj_path, 
            debugging, debug_freq,
            occlusion,
            policy_path
            # <<<< [YC] add
            ):
    
    
    # --------------------------- Warm Up Stage -------------------------- #
    
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = gaussianModel[gs_type](dataset.sh_degree) # [YC] note: nothing changing here
    print("Training func policy_path:", policy_path)
        
    # >>>> [YC] add: if there is textured mesh, load it here (before training loop)
    textured_mesh = None
    if texture_obj_path != "":
        print("Loading textured mesh for background rendering...")
        textured_mesh = load_objs_as_meshes([texture_obj_path]).to("cuda") # [YC] add
    # <<<< [YC] add
    
    
    #! [YC] note: main changing point is here
    scene = Scene(dataset, gaussians, policy_path=policy_path, texture_obj_path=texture_obj_path)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    if debugging:
        print("Debugging mode is on.")
        check_path = Path(scene.model_path)/"debugging"/"training_check"
        check_path.mkdir(parents=True, exist_ok=True)
    
    
    # [NOTE] workaround
    # warmup(scene.getTrainCameras().copy(), textured_mesh)
      
    if dataset.warmup_only:
        print("[INFO] Only run warmup stage, exiting...")
        exit()
    
    
    print("[INFO] Finished Warm-Up, Start Training..." )
    
    # Not sure why need to get background in this part
    # --------------------------- Load background image -------------------------- #
    background_image_path = "/mnt/data1/syjintw/NEU/dataset/hotdog/mesh_texture/r_0.png"
    img = Image.open(background_image_path).convert("RGB")
    viewpoint_camera_height = 800
    viewpoint_camera_width = 800
    img = img.resize((viewpoint_camera_height, viewpoint_camera_width), Image.BILINEAR)
    transform = T.Compose([
        T.ToTensor(),  # [0, 255] → [0.0, 1.0], shape (3, H, W)
    ])
    background = transform(img).to(torch.float32).cuda()
    
    # ----------------------------- Load depth image ----------------------------- #
    background_depth_pt_path = "/mnt/data1/syjintw/NEU/dataset/hotdog/mesh_depth/r_0.pt"
    background_depth = torch.load(background_depth_pt_path).unsqueeze(0)
    # <<<< [YC]

    # ---------------------------------------------------------------------------- #
    #                              Start Training Loop                             #
    # ---------------------------------------------------------------------------- #
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        os.makedirs(f"{scene.model_path}/xyz", exist_ok=True)
        if save_xyz and (iteration % 5000 == 1 or iteration == opt.iterations):
            torch.save(gaussians.get_xyz, f"{scene.model_path}/xyz/{iteration}.pt")
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    # net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image = render(custom_cam, gaussians, pipe, background, background_depth, scaling_modifer)["render"] # [YC] add
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        rand_cam_id = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_cam_id)
        
        # ---------------------------------------------------------------------------- #
        #                                Load Background                               #
        # ---------------------------------------------------------------------------- #
        viewpoint_camera_height = viewpoint_cam.image_height
        viewpoint_camera_width = viewpoint_cam.image_width

        transform = T.Compose([
            T.ToTensor(),  # [0, 255] → [0.0, 1.0], shape (3, H, W)
        ])
        
        # ------------------------------ Mesh background ----------------------------- #
        bg = None
        if textured_mesh is None:
            # print("Loading precaptured background for rendering...")
            bg_image_path = f"/mnt/data1/syjintw/NEU/dataset/hotdog/mesh_texture/{viewpoint_cam.image_name}.png"
            img = Image.open(bg_image_path).convert("RGB")
            img = img.resize((viewpoint_camera_width, viewpoint_camera_height), Image.BILINEAR) # (W, H)
            bg = transform(img).to(torch.float32).cuda()
        
        # ------------------------------ Mesh depth background ----------------------------- #
        bg_depth = None
        if textured_mesh is None:
            # print("Loading precaptured depth for rendering...")
            bg_depth_pt_path = f"/mnt/data1/syjintw/NEU/dataset/hotdog/mesh_depth/{viewpoint_cam.image_name}.pt"
            bg_depth = torch.load(bg_depth_pt_path).unsqueeze(0).to("cuda")

        # ------------------------------ Pure background ----------------------------- #
        pure_bg_template = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        pure_bg = torch.tensor(pure_bg_template, dtype=torch.float32, device="cuda").view(3, 1, 1)
        pure_bg = pure_bg.expand(3, viewpoint_camera_height, viewpoint_camera_width)
        
        # --------------------- Pure depth background (all zeros) -------------------- #
        pure_bg_depth = torch.full((1, viewpoint_camera_height, viewpoint_camera_width), 0, dtype=torch.float32, device="cuda")
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # >>>> [YC]
        # -------------------------- Rendering for training -------------------------- #
        if gs_type == "gs":
            render_pkg = render(viewpoint_cam, gaussians, pipe, 
                                bg_color=pure_bg, bg_depth=pure_bg_depth)
        elif gs_type == "gs_mesh":
            if occlusion: # [YC] use occlusion diff-gaussian-rasterizer for training
                render_pkg = render(viewpoint_cam, gaussians, pipe, 
                                    bg_color=bg, bg_depth=bg_depth,
                                    textured_mesh=textured_mesh) # [YC] if there is textured mesh, it will use mesh renderer to get bg and bg_depth
                print("[INFO] train:: using occlusion-handling rasterizer for gs_mesh")
                
            else: # [YC] use original diff-gaussian-rasterizer for training
                render_pkg = render(viewpoint_cam, gaussians, pipe, 
                                    bg_color=bg, bg_depth=pure_bg_depth,
                                    textured_mesh=textured_mesh) # [YC] no occlusion handling, always use pure bg and pure depth
                print("[INFO] train:: using vanilla rasterizer for gs_mesh")
                
                
        # render_pkg = render(viewpoint_cam, gaussians, pipe, bg, bg_depth)
        
        image = render_pkg["render"]
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # -------------------------- Load ground truth image ------------------------- #
        gt_image = viewpoint_cam.original_image.cuda()

        if debugging:
            # ------------------- Change Tensor to PIL.Image for saving ------------------ #
            if iteration % debug_freq == 0:
                # ---------------------------- Ground truth image ---------------------------- #
                gt_img_to_save = gt_image.detach().clamp(0, 1).cpu()
                gt_img_pil = TF.to_pil_image(gt_img_to_save)
                gt_img_pil.save(check_path/f"{iteration}_gt.png")
                
                # ------------------------ Render image from training ------------------------ #
                img_to_save = image.detach().clamp(0, 1).cpu()
                img_pil = TF.to_pil_image(img_to_save)
                img_pil.save(check_path/f"{iteration}_training.png")
                
                # ----------------------- Background image for training ---------------------- #
                img_to_save = render_pkg["bg_color"].detach().clamp(0, 1).cpu()
                img_pil = TF.to_pil_image(img_to_save)
                img_pil.save(check_path/f"{iteration}_training_bg.png")
                
                if gs_type == "gs_mesh":
                    # ------------- Render mesh background and depth background ------------- #
                    # [1, 1, 1]
                    render_mesh_with_depth = render(viewpoint_cam, gaussians, pipe, 
                                                    bg_color=bg, bg_depth=bg_depth,
                                                    textured_mesh=textured_mesh)
                    _image = render_mesh_with_depth["render"]

                    img_to_save = _image.detach().clamp(0, 1).cpu()
                    img_pil = TF.to_pil_image(img_to_save)
                    img_pil.save(check_path/f"{iteration}_gs_mesh_with_depth.png")
                
                    # ------------- Render mesh background and fake depth background ------------- #
                    # [0, 1, 1]
                    render_mesh_wo_depth = render(viewpoint_cam, gaussians, pipe, 
                                                    bg_color=bg, bg_depth=pure_bg_depth,
                                                    textured_mesh=textured_mesh)
                    _image = render_mesh_wo_depth["render"]

                    img_to_save = _image.detach().clamp(0, 1).cpu()
                    img_pil = TF.to_pil_image(img_to_save)
                    img_pil.save(check_path/f"{iteration}_gs_mesh_wo_depth.png")

                    # ------------- Render pure background and mesh depth background ------------- #
                    # [1, 0, 1]
                    render_pure_with_depth = render(viewpoint_cam, gaussians, pipe, 
                                                    bg_color=pure_bg, bg_depth=bg_depth,
                                                    textured_mesh=textured_mesh)
                    _image = render_pure_with_depth["render"]
                    
                    img_to_save = _image.detach().clamp(0, 1).cpu()
                    img_pil = TF.to_pil_image(img_to_save)
                    img_pil.save(check_path/f"{iteration}_gs_pure_with_depth.png")
                
                    # ------------- Render pure background and fake depth background ------------- #
                    # [1, 1, 1]
                    render_pure_wo_depth = render(viewpoint_cam, gaussians, pipe, 
                                                bg_color=pure_bg, bg_depth=pure_bg_depth,
                                                textured_mesh=None)
                    _image = render_pure_wo_depth["render"]
                    
                    img_to_save = _image.detach().clamp(0, 1).cpu()
                    img_pil = TF.to_pil_image(img_to_save)
                    img_pil.save(check_path/f"{iteration}_gs_pure_wo_depth.png")
            # <<<< [YC]
            
        # Compute loss and backpropagate
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # # Brutally adjust loss, but keeping the backward information
        # Ll1 = 0.0
        # loss = image.mean() * 0.0 + 0.5

        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
            #                 testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            #! [YC] note: original "gs_mesh" will skip densification
            # Densification
            if (args.gs_type == "gs") or (args.gs_type == "gs_flat"):
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                    size_threshold)

                    if iteration % opt.opacity_reset_interval == 0 or (
                            dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()
            # >>>> [YC] add
            elif args.gs_type == "gs_mesh":
                pass
            # <<<< [YC] add

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if hasattr(gaussians, 'update_alpha'):
            gaussians.update_alpha()
        if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("[INFO] Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("[INFO] Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--gs_type', type=str, default="gs_mesh")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--meshes", nargs="+", type=str, default=[])
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--save_xyz", action='store_true')
    
    # >>>> [YC] add
    parser.add_argument('--texture_obj_path', type=str, default="")
    parser.add_argument('--debugging', action='store_true')
    parser.add_argument('--debug_freq', type=int, default=1, help="Iteration of saving debugging images")
    parser.add_argument('--occlusion', action='store_true')
    parser.add_argument('--policy_path', type=str, default="", 
        help="Path to the pre-computed .npy file storing num_gs_per_tri[]. \
            When this is provided, it has higher priority than --alloc_policy; \
            otherwise, will overwrite/recompute")
    # parser.add_argument('--precaptured_mesh_img_path', type=str, default="")
    # <<<< [YC] add
    
    parser.add_argument("--total_splats", type=int, default=131_072, help="Total number of splats to allocate (default: 2^17=131072)")
    parser.add_argument("--alloc_policy", type=str, default="area", help="Allocation policy for splats (default: area)")
    parser.add_argument("--warmup_only", action='store_true', help="only run warmup stage and exit, no entering training loop")
    
    
    

    lp = ModelParams(parser)
    args, _ = parser.parse_known_args(sys.argv[1:])
    lp.num_splats = args.num_splats
    lp.meshes = args.meshes
    lp.gs_type = args.gs_type
    
    # >>>> [Sam] add
    lp.total_splats = args.total_splats
    lp.alloc_policy = args.alloc_policy 
    lp.warmup_only = args.warmup_only
    # <<<< [Sam] add

    op = optimizationParamTypeCallbacks[args.gs_type](parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)

    print("torch cuda: ", torch.cuda.is_available())
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        args.gs_type,
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint, args.debug_from, args.save_xyz,
        # >>>> [YC] add
        texture_obj_path=args.texture_obj_path,
        debugging=args.debugging, debug_freq=args.debug_freq,
        occlusion=args.occlusion,
        policy_path=args.policy_path
        # <<<< [YC] add
    )

    # All done
    print("\n[INFO] Training complete.")
