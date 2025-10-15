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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import trimesh

from pytorch3d.renderer import (
    AmbientLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    )
from pytorch3d.renderer.blending import BlendParams
from torchvision.transforms import functional as TF, InterpolationMode
from scene.cameras import convert_camera_from_gs_to_pytorch3d
from PIL import Image

import time

def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices

def render(viewpoint_camera, pc : GaussianModel, pipe, 
           bg_color : torch.Tensor, bg_depth : torch.Tensor,
           scaling_modifier = 1.0, override_color = None,
           textured_mesh=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    
    # >>>> [YC] To support real time mesh rendering as background
    if textured_mesh is not None:
        start_time = time.time()
        image_height = 800
        image_width = 800
        faces_per_pixel = 1
        
        p3d_cameras = convert_camera_from_gs_to_pytorch3d(
            [viewpoint_camera]
        )
        # print(len(p3d_cameras))
        
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
        
        # ------------------------------- Handle color ------------------------------- #
        with torch.no_grad():
            rgb_img = renderer(textured_mesh, cameras=p3d_cameras)[0, ..., :3]
        
        bg_color = rgb_img.permute(2, 0, 1).contiguous()
        
        # ------------------------- Save color for debugging ------------------------- #
        # bg_pil = TF.to_pil_image(bg_color.cpu())   # Convert tensor → PIL Image
        # bg_pil.save("./bg_color.png")
        
        # ------------------------------- Handle depth ------------------------------- #
        # Get fragments from rasterizer
        fragments = rasterizer(textured_mesh, cameras=p3d_cameras)
        # Nearest surface depth in NDC space
        depth = fragments.zbuf[0, ..., 0]  # (H, W)
        # Mask out pixels that didn’t hit any face
        mask = fragments.pix_to_face[0, ..., 0] >= 0
        depth = depth.masked_fill(~mask, -1)
        
        # print(depth.unsqueeze(0).shape, bg_depth.shape)
        # print(type(depth.unsqueeze(0)), type(bg_depth))
        # are_equal = torch.equal(depth.unsqueeze(0), bg_depth)
        # print("Exact match:", are_equal)        
        
        bg_depth = depth.unsqueeze(0)
        
        # ------------------------ Save depth pt for debugging ----------------------- #
        # torch.save(depth, "./bg_depth.pt")
        
        # ---------------------- Save depth image for debugging ---------------------- #
        # # Replace NaNs with max depth for visualization
        # valid_depth = depth[~torch.isnan(depth)]
        # if valid_depth.numel() > 0:
        #     max_val = valid_depth.max()
        # else:
        #     max_val = 0.0
        # depth_vis = depth.clone()
        # depth_vis[torch.isnan(depth_vis)] = max_val
        # # Normalize depth to 0–1 for image saving
        # min_val = depth_vis.min()
        # max_val = depth_vis.max()
        # # print(f"Depth min: {min_val}, max: {max_val}")
        # depth_vis = (depth_vis - min_val) / (max_val - min_val + 1e-8)
        # # Add channel, resize to exactly 800x800, then drop channel
        # depth_resized = TF.resize(
        #     depth_vis.unsqueeze(0),  # (1,H,W)
        #     [800, 800],
        #     interpolation=InterpolationMode.BILINEAR,
        #     antialias=True
        # ).squeeze(0)  # (800,800)
        # # Convert to 8-bit grayscale and save
        # depth_u8 = (depth_resized * 255).round().clamp(0, 255).to(torch.uint8)  # (800,800)
        # depth_pil = Image.fromarray(depth_u8.cpu().numpy(), mode='L')
        # # Convert to PIL Image and save as PNG
        # depth_pil.save("./bg_depth.png")
        
        end_time = time.time()
        # print(f"Mesh rendering time: {end_time - start_time:.4f} seconds")
    # <<<< [YC]
        
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # Set up rasterization configuration
    # print("viewpoint_camera.FoVx:", viewpoint_camera.FoVx)
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    viewpoint_camera.camera_center = viewpoint_camera.camera_center
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        depth=bg_depth # [YC]
        # antialiasing=pipe.antialiasing
    )

    # # For debugging:
    # print("tanfovx:", tanfovx, "tanfovy:", tanfovy)
    # print("Camera center:", viewpoint_camera.camera_center)
    # print("scaling_modifier:", scaling_modifier)
    
    # print("viewpoint_camera.world_view_transform:", viewpoint_camera.world_view_transform)
    # print("viewpoint_camera.full_proj_transform:", viewpoint_camera.full_proj_transform)
    
    # print("GaussianRasterizer:init")
    rasterizer = GaussianRasterizer(raster_settings=raster_settings) # __init__
    
    _xyz = pc.get_xyz

    means3D = _xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # rendered_image, radii, depth_image = rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp) # forward

    # print("rasterizer")
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp) # forward
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            # "depth": depth_image
            }
