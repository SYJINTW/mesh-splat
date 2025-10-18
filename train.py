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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from renderer.gaussian_renderer import render, network_gui
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

SCENE_NAME = "hotdog"
check_path = Path(f'/mnt/data1/syjintw/NEU/mesh-splat/training_check/{SCENE_NAME}')
check_path.mkdir(parents=True, exist_ok=True)

def warmup(viewpoint_cameras):
    """
    Render the point cloud using cameras from 3DGS
    """
    print("Warmup rendering...")
    p3d_cameras = convert_camera_from_gs_to_pytorch3d(
        viewpoint_cameras
    )
    
    pcd = o3d.io.read_point_cloud("/mnt/data1/syjintw/NEU/dataset/hotdog/points3d.ply")
    points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
    colors = torch.tensor(np.asarray(pcd.colors), dtype=torch.float32, device='cuda')
    point_cloud = Pointclouds(points=[points], features=[colors])
    
    raster_settings = PointsRasterizationSettings(
        image_size=(800, 800),
        radius=0.001,        # controls point size
        points_per_pixel=10 # controls density
    )
    
    for idx, p3d_camera in enumerate(p3d_cameras):
        lights = AmbientLights(device="cuda")
        rasterizer = PointsRasterizer(
            cameras=p3d_camera, 
            raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor()
        )
        
        image = renderer(point_cloud)  # (1, H, W, 4)
        image = image[0, ..., :3]
        image_color = image.permute(2, 0, 1).contiguous()
        
        image_pil = TF.to_pil_image(image_color.cpu())   # Convert tensor → PIL Image
        image_pil.save(f"./warmup/{idx}.png")

def warmup_new(viewpoint_cameras, p3d_mesh):
    """
    Render with mesh
    """
    # Trimesh
    tm_mesh = trimesh.load("/mnt/data1/syjintw/NEU/dataset/hotdog/mesh.obj", process=False)
    tm_faces = tm_mesh.faces  # (F, 3)
    # tm_faces_sorted = np.sort(tm_faces, axis=1)  # sort each row (triangle)
    
    # PyTorch3D
    verts = torch.tensor(tm_mesh.vertices, dtype=torch.float32, device="cuda")
    faces = torch.tensor(tm_mesh.faces, dtype=torch.int64, device="cuda")  # keep order!
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb)
    p3d_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    
    image_height = 800
    image_width = 800
    faces_per_pixel = 1
    
    triangle_filter = np.zeros(faces.shape[0], dtype=bool)
    
    filters = []
    for i in range(len(viewpoint_cameras)):
        p3d_cameras = convert_camera_from_gs_to_pytorch3d(
            [viewpoint_cameras[i]]
        )
        
        mesh_raster_settings = RasterizationSettings(
            image_size=(image_height, image_width),
            blur_radius=0.0, 
            faces_per_pixel=faces_per_pixel,
            # max_faces_per_bin=max_faces_per_bin
        )
        lights = AmbientLights(device="cuda")
        
        rasterizer = MeshRasterizer(
            cameras=p3d_cameras[0], 
            raster_settings=mesh_raster_settings,
        )
        
        fragments = rasterizer(p3d_mesh)
        
        renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=SoftPhongShader(
                device="cuda", 
                cameras=p3d_cameras[0],
                lights=lights,
                # blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)),
                blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),
            )
        )
        
        rgb_img = renderer(p3d_mesh, cameras=p3d_cameras)[0, ..., :3]
        face_idx_map = fragments.pix_to_face[0, ..., 0]
        # print(rgb_img.shape, face_idx_map.shape)
        
        # mesh_scene = trimesh.load(f'/mnt/data1/syjintw/NEU/dataset/hotdog/mesh.obj', force='mesh')
        # mesh_scene.apply_transform(trimesh.transformations.rotation_matrix(
        #     angle=-np.pi/2, direction=[1, 0, 0], point=[0, 0, 0]
        # ))
        
        # ------------------ Load mask ------------------
        mask_img = Image.open(f"/mnt/data1/syjintw/NEU/dataset/hotdog/distortion_thres0.05/r_{i}.png").convert("L")  # grayscale
        mask = np.array(mask_img)
        mask_bool = mask > 128  # white = keep (True), black = drop (False)
        H, W = mask_bool.shape
        
        # ------------------ Face index per pixel ------------------
        face_idx_map = fragments.pix_to_face[0, ..., 0].cpu().numpy()  # (H, W)
        # face_idx_map = -1 means background
        
        # ------------------ Determine which faces to keep ------------------
        face_keep = np.zeros(p3d_mesh.num_faces_per_mesh()[0].item(), dtype=bool)
        valid_mask = face_idx_map >= 0
        # faces that map to white pixels
        white_faces = face_idx_map[valid_mask & mask_bool]
        face_keep[np.unique(white_faces)] = True
        
        # ------------------ Extract mesh geometry ------------------
        verts = p3d_mesh.verts_packed().cpu().numpy()
        faces = p3d_mesh.faces_packed().cpu().numpy()

        kept_faces = faces[face_keep]
        kept_face_idx = np.where(face_keep)[0]

        # print("faces.shape:", faces.shape)
        # print("kept_faces.shape:", kept_faces.shape)
        # print("kept_faces:", kept_faces)
        # print("kept_face_idx:", kept_face_idx)

        # triangle_filter = np.zeros(faces.shape[0], dtype=bool)
        triangle_filter[kept_face_idx] = 1
        
        # # Remove unreferenced vertices
        # unique_verts_idx, new_faces_idx = np.unique(kept_faces.flatten(), return_inverse=True)
        # kept_verts = verts[unique_verts_idx]
        # new_faces = new_faces_idx.reshape(kept_faces.shape)

        # # ------------------ Export with Open3D ------------------
        # mesh_o3d = o3d.geometry.TriangleMesh()
        # mesh_o3d.vertices = o3d.utility.Vector3dVector(kept_verts)
        # mesh_o3d.triangles = o3d.utility.Vector3iVector(new_faces)
        # mesh_o3d.compute_vertex_normals()
        # o3d.io.write_triangle_mesh("../r_0.obj", mesh_o3d)
        # print(f"[✔] Exported filtered mesh with {len(kept_verts)} verts and {len(new_faces)} faces to ../r_0.obj")
    
    print("Saving triangle filter...")
    print(triangle_filter.shape, triangle_filter.sum())
    np.save("../triangle_filter.npy", triangle_filter)

def test():
    # Trimesh
    tm_mesh = trimesh.load("/mnt/data1/syjintw/NEU/dataset/hotdog/mesh.obj", process=False)
    tm_faces = tm_mesh.faces  # (F, 3)
    # tm_faces_sorted = np.sort(tm_faces, axis=1)  # sort each row (triangle)
    
    # PyTorch3D
    verts = torch.tensor(tm_mesh.vertices, dtype=torch.float32, device="cuda")
    faces = torch.tensor(tm_mesh.faces, dtype=torch.int64, device="cuda")  # keep order!
    p3d_mesh = Meshes(verts=[verts], faces=[faces])
    # p3d_mesh = load_objs_as_meshes(["/mnt/data1/syjintw/NEU/dataset/hotdog/mesh.obj"]).to("cuda")

    # Access vertices and faces explicitly if needed:
    p3d_verts = p3d_mesh.verts_list()[0]  # (V, 3)
    p3d_faces = p3d_mesh.faces_list()[0].cpu().numpy()  # (F, 3)
    # p3d_faces_sorted = np.sort(p3d_faces, axis=1)

    print("Same number of faces?", tm_faces.shape[0] == p3d_faces.shape[0])
    print("First 5 trimesh faces:\n", tm_faces[:5])
    print("First 5 p3d faces:\n", p3d_faces[:5])
    # print("First 5 trimesh faces:\n", tm_faces_sorted[:5])
    # print("First 5 p3d faces:\n", p3d_faces_sorted[:5])
    
    target = tm_faces[0]
    exists = np.any(np.all(p3d_faces == target, axis=1))
    # target = tm_faces_sorted[0]
    # exists = np.any(np.all(p3d_faces_sorted == target, axis=1))
    
    print("Exists?", exists)

def training(gs_type, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, save_xyz):
    # test()
    # exit()
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = gaussianModel[gs_type](dataset.sh_degree)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    textured_mesh = load_objs_as_meshes(["/mnt/data1/syjintw/NEU/dataset/hotdog/mesh.obj"]).to("cuda")
    
    # warmup(scene.getTrainCameras().copy())
    # warmup_new(scene.getTrainCameras().copy(), textured_mesh)
    # exit()
    
    # >>>> [YC] 
    # Change diff-rasterization settings

    # bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # # Load static image background
    # background_image_path = "/home/syjintw/Desktop/NEU/gaussian-mesh-splatting/output_render_no_light.png"
    # img = Image.open(background_image_path).convert("RGB")
    # viewpoint_camera_height = 800
    # viewpoint_camera_width = 800
    # # === 1. Use PIL to load images ===
    # img = img.resize((viewpoint_camera_height, viewpoint_camera_width), Image.BILINEAR)
    # # === 2. Change image to Tensor ===
    # transform = T.Compose([
    #     T.ToTensor(),  # [0, 255] → [0.0, 1.0], shape (3, H, W)
    # ])
    # background = transform(img).to(torch.float32).cuda()
    
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
        # rand_cam_id = 0
        viewpoint_cam = viewpoint_stack.pop(rand_cam_id)
        # print(viewpoint_cam.uid, viewpoint_cam.image_name)
        
        # >>>> [YC] 
        # ------------------------------ Mesh background ----------------------------- #
        background_image_path = f"/mnt/data1/syjintw/NEU/dataset/hotdog/mesh_texture/{viewpoint_cam.image_name}.png"
        img = Image.open(background_image_path).convert("RGB")
        viewpoint_camera_height = 800
        viewpoint_camera_width = 800
        img = img.resize((viewpoint_camera_width, viewpoint_camera_height), Image.BILINEAR) # (W, H)

        if iteration % 100 == 0:
            output_path = "output.png"
            img.save(output_path, format="PNG")

        transform = T.Compose([
            T.ToTensor(),  # [0, 255] → [0.0, 1.0], shape (3, H, W)
        ])
        background = transform(img).to(torch.float32).cuda()
        
        # ------------------------------ Mesh depth background ----------------------------- #
        background_depth_pt_path = f"/mnt/data1/syjintw/NEU/dataset/hotdog/mesh_depth/{viewpoint_cam.image_name}.pt"
        background_depth = torch.load(background_depth_pt_path)

        # print("background_depth.shape:", background_depth.shape)
        # print("background_depth[:][350][350]:", background_depth[0][350][350])
        # print("background_depth[:][400][400]:", background_depth[0][400][400])
        
        # bg = torch.rand((3), device="cuda") if opt.random_background else background
        # bg_depth = torch.rand((3), device="cuda") if opt.random_background else background_depth
        bg = background
        bg_depth = background_depth.unsqueeze(0).to("cuda")
        
        # ------------------------------ Pure background ----------------------------- #
        viewpoint_camera_height = 800
        viewpoint_camera_width = 800
        
        pure_bg_template = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        # pure_bg_template = [0, 0, 0]
        pure_bg = torch.tensor(pure_bg_template, dtype=torch.float32, device="cuda")
        pure_bg_depth = torch.full((1, viewpoint_camera_height, viewpoint_camera_width), 0, dtype=torch.float32, device="cuda")
        # <<<< [YC] 
    
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # >>>> [YC]
        # ------------- Render mesh background and mesh depth background ------------- #
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, bg_depth,
                            textured_mesh=textured_mesh)
        # render_pkg = render(viewpoint_cam, gaussians, pipe, bg, pure_bg_depth)
        image = render_pkg["render"]
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # ------------- Render pure background and fake depth background ------------- #
        render_pure_bg = render(viewpoint_cam, gaussians, pipe, pure_bg, pure_bg_depth)
        image_gs = render_pure_bg["render"]
        # viewspace_point_tensor, visibility_filter, radii = render_pure_bg["viewspace_points"], render_pure_bg["visibility_filter"], render_pure_bg["radii"]

        # ------------- Render pure mesh background and mesh depth background ------------- #
        render_pure_mesh_bg = render(viewpoint_cam, gaussians, pipe, pure_bg, bg_depth)
        image_gs_mesh = render_pure_mesh_bg["render"]
        
        # -------------------------- Load ground truth image ------------------------- #
        gt_image = viewpoint_cam.original_image.cuda()

        # ------------------- Change Tensor to PIL.Image for saving ------------------ #
        if iteration % 100 == 0:
            img_to_save = image.detach().clamp(0, 1).cpu()
            img_pil = TF.to_pil_image(img_to_save)
            img_pil.save(check_path/f"{iteration}.png")
            
            gs_img_to_save = image_gs.detach().clamp(0, 1).cpu()
            gs_img_pil = TF.to_pil_image(gs_img_to_save)
            gs_img_pil.save(check_path/f"{iteration}_gs.png")
            
            gs_mesh_img_to_save = image_gs_mesh.detach().clamp(0, 1).cpu()
            gs_mesh_img_pil = TF.to_pil_image(gs_mesh_img_to_save)
            gs_mesh_img_pil.save(check_path/f"{iteration}_gs_mesh.png")
            
            gt_img_to_save = gt_image.detach().clamp(0, 1).cpu()
            gt_img_pil = TF.to_pil_image(gt_img_to_save)
            gt_img_pil.save(check_path/f"{iteration}_gt.png")

        # <<<< [YC]
        
        # Loss
        # Ll1 = l1_loss(image, gt_image)
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_gs, gt_image))
        
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
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
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

    lp = ModelParams(parser)
    args, _ = parser.parse_known_args(sys.argv[1:])
    lp.num_splats = args.num_splats
    lp.meshes = args.meshes
    lp.gs_type = args.gs_type

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
        args.start_checkpoint, args.debug_from, args.save_xyz
    )

    # All done
    print("\nTraining complete.")
