#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use
#

import os
import numpy as np
import trimesh
import torch

from games.mesh_splatting.utils.graphics_utils import MeshPointCloud
from scene.dataset_readers import (
    readColmapSceneInfo,
    readNerfSyntheticInfo,
    readCamerasFromTransforms,
    getNerfppNorm,
    SceneInfo,
    storePly,
)
from utils.sh_utils import SH2RGB
from scene.budgeting import get_budgeting_policy

from pathlib import Path

softmax = torch.nn.Softmax(dim=2)


def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices


def readNerfSyntheticMeshInfo(
        path, white_background, eval, num_splats, extension=".png",
        # >>>> [YC] add
        texture_obj_path: str = None,
        policy_path: str = None,
        # <<<< [YC] add
        # >>>> [SAM] add budgeting policy params
        total_splats: int = None,  # if None, falls back to num_splats per triangle
        budgeting_policy_name: str = "uniform",
        min_splats_per_tri: int = 0,
        max_splats_per_tri: int = 8,
        # <<<< [SAM] add
) -> SceneInfo:
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if texture_obj_path is None:
        print(f"[INFO] DatasetReader::Reading Mesh object from {path}/mesh.obj")
        mesh_scene = trimesh.load(f'{path}/mesh.obj', force='mesh')
    else:
        print(f"Reading Mesh object from {texture_obj_path}")
        mesh_scene = trimesh.load(texture_obj_path, force='mesh')

    # >>>> [YC] add: because the mesh is generated from torch3d, so need to rotate
    mesh_scene.apply_transform(trimesh.transformations.rotation_matrix(
        angle=-np.pi/2, direction=[1, 0, 0], point=[0, 0, 0]
    ))
    # <<<< [YC] add
    
    vertices = mesh_scene.vertices
    vertices = transform_vertices_function(
        torch.tensor(vertices),
    )
    faces = mesh_scene.faces
    triangles = vertices[torch.tensor(mesh_scene.faces).long()].float()
    
    # >>>> [YC] add
    # Access texture info
    print("type(mesh_scene.visual):", type(mesh_scene.visual)) # [YC] debug: should be TextureVisuals
    print("mesh_scene.visual.uv.shape:", mesh_scene.visual.uv.shape) # [YC] debug: UV coordinates per vertex
    print("mesh_scene.visual.material.image:", mesh_scene.visual.material.image) # [YC] debug: PIL image
    
    # Extract texture as numpy
    texture_img = np.array(mesh_scene.visual.material.image)  # (H, W, 3) or (H, W, 4)
    H, W = texture_img.shape[:2]
    
    # Get per-vertex UVs
    uv_coords = mesh_scene.visual.uv  # (n_vertices, 2), values in [0,1]
    
    # Face UVs (map vertex index -> uv)
    face_uvs = uv_coords[faces]  # (n_faces, 3, 2)
    # <<<< [YC] add
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    print("ply_path:", ply_path)
    
    # if not os.path.exists(ply_path):
    if True:
        # >>>> [SAM] Budgeting policy integration
        # Determine allocation strategy
        if policy_path is not None and policy_path != "":
            # [TODO]: load from file
            # [TODO] integrate error-map-based splat allocation
            
            allocation_path = Path(policy_path) # path to the .npy file storing splat allocation
            # num_splats[]
            # scene_name
            # min, max
            
            if allocation_path.exists(): 
                print("Loading splat allocation from:", allocation_path)
                num_splats_per_triangle = np.full(triangles.shape[0], 1, dtype=int)
                print("Initial max and min:", num_splats_per_triangle.max(), num_splats_per_triangle.min())
                num_splats_per_triangle = np.load(allocation_path)
                print("Final Max and min:", num_splats_per_triangle.max(), num_splats_per_triangle.min())
            else:
                print("No splat allocation found at:", allocation_path)
                num_splats_per_triangle = np.full(triangles.shape[0], num_splats, dtype=int)
                
                
        elif total_splats is not None:
            # Use budgeting policy
            print(f"[INFO] Scene::Using budgeting policy: {budgeting_policy_name}")
            budgeting_policy = get_budgeting_policy(
                budgeting_policy_name, 
                mesh=mesh_scene,
                viewpoint_camera_infos=train_cam_infos,
                dataset_path=path,
                )
            
            num_splats_per_triangle = budgeting_policy.allocate(
                triangles=triangles,
                total_splats=total_splats,
                min_per_tri=min_splats_per_tri,
                max_per_tri=max_splats_per_tri,
            )
            
            print(f"[INFO] Scene::Requested total splats: {total_splats}")
            print(f"[INFO] Scene::Allocated total splats: {num_splats_per_triangle.sum()}")
            print(f"[INFO] Scene::Min/Max splats per triangle: {num_splats_per_triangle.min()}/{num_splats_per_triangle.max()}")
            print(f"[INFO] Scene::Mean/Std splats per triangle: {num_splats_per_triangle.mean():.2f}/{num_splats_per_triangle.std():.2f}")
            
            
            # under {dataset_path}/policy
            allocation_save_path = Path(path)/ f"policy/{budgeting_policy_name}_{total_splats}.npy"
            print(f"[INFO] Scene::Saving splat allocation to: {allocation_save_path}")
            np.save(allocation_save_path, num_splats_per_triangle)
            
        else:
            # Default: uniform distribution using num_splats
            num_splats_per_triangle = np.full(triangles.shape[0], num_splats, dtype=int)
            print(f"[INFO] Scene::Fallback using uniform distribution: {num_splats} splats per triangle")
        # <<<< [SAM] Budgeting policy integration
        
        # Since this data set has no colmap data, we start with random points
        num_pts = num_splats_per_triangle.sum()
        print(f"Generating random point cloud ({num_pts})...")
        print(f"Average points per triangle: {num_pts / triangles.shape[0] if triangles.shape[0] > 0 else 0}...")
        
        # ---------------------------------------------------------------------------- #
        #                 Get initial Gaussian colors from texture map                 #
        # ---------------------------------------------------------------------------- #
        face_uvs = mesh_scene.visual.uv[faces]  # (n_faces, 3, 2)
        H, W = texture_img.shape[:2]        
        
        # Convert UVs (3 per face) to pixel coordinates
        px = (face_uvs[..., 0] * (W - 1)).astype(int)
        py = ((1 - face_uvs[..., 1]) * (H - 1)).astype(int)
        
        # Clamp
        px = np.clip(px, 0, W - 1)
        py = np.clip(py, 0, H - 1)
        
        # Sample 3 vertex colors per triangle
        tri_vertex_colors = texture_img[py, px, :3]   # (n_faces, 3, 3)
        
        # Average per-triangle color
        tri_avg_colors = tri_vertex_colors.mean(axis=1)  # (n_faces, 3)
        
        print("tri_avg_colors:", tri_avg_colors.shape) # [YC] debug
        # colors = np.repeat(tri_avg_colors, num_splats, axis=0)  # (num_pts, 3)
        
        # We create random points inside the bounds triangles
        xyz_list = []
        alpha_list = []
        color_list = []
        tri_indices_list = []
        
        # >>>> [SAM] Build point-to-triangle mapping
        for i in range(triangles.shape[0]):
            n = num_splats_per_triangle[i]
            if n == 0:
                continue
                
            alpha = torch.rand(n, 3)
            alpha = alpha / alpha.sum(dim=1, keepdim=True)  # normalize to barycentric coords

            pts = (alpha[:, 0:1] * triangles[i, 0] +
                alpha[:, 1:2] * triangles[i, 1] +
                alpha[:, 2:3] * triangles[i, 2])

            color = np.repeat(tri_avg_colors[i].reshape(1, 3), n, axis=0)  # (num_pts, 3)
            # print(color.shape) # [YC] debug
            
            xyz_list.append(pts)
            alpha_list.append(alpha)
            color_list.append(color)
            tri_indices_list.append(torch.full((n,), i, dtype=torch.long))
        # <<<< [SAM]
        
        xyz = torch.cat(xyz_list, dim=0)
        xyz = xyz.reshape(num_pts, 3)
        
        alpha = torch.cat(alpha_list, dim=0)
        
        # shs = np.random.random((num_pts, 3)) / 255.0
        colors = np.concatenate(color_list, axis=0)
        print(colors.shape, xyz.shape, alpha.shape) # [YC] debug
        
        points = trimesh.points.PointCloud(np.array(xyz))

        # Combine into a scene
        scene = trimesh.Scene()
        scene.add_geometry(mesh_scene)
        scene.add_geometry(points)
        
        tri_indices = torch.cat(tri_indices_list, dim=0)
        
        pcd = MeshPointCloud(
            alpha=alpha,
            points=xyz,
            # colors=SH2RGB(shs),
            colors=colors/255.0,
            normals=np.zeros((num_pts, 3)),
            vertices=vertices,
            faces=faces,
            transform_vertices_function=transform_vertices_function,
            triangles=triangles.cuda(),
            triangle_indices=tri_indices
        )
        print("Created MeshPointCloud with", pcd.points.shape[0], "points.")

        # storePly(ply_path, pcd.points, SH2RGB(shs) * 255)
        # storePly(ply_path, pcd.points, colors)

    print(f"[DEBUG] DatasetReader:: ply_path={ply_path}")
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Blender_Mesh": readNerfSyntheticMeshInfo
}
