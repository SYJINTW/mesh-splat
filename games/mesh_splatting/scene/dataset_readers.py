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

from pathlib import Path

softmax = torch.nn.Softmax(dim=2)


def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices


def readNerfSyntheticMeshInfo(
        path, white_background, eval, num_splats, extension=".png"
):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    print("Reading Mesh object")
    mesh_scene = trimesh.load(f'{path}/mesh.obj', force='mesh')
    
    # [NOTE] trimesh and pytorch3d have different coordinate system
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
        
        # Non-uniform splatting density based on triangle area
        # areas = trimesh.triangles.area(triangles.cpu().numpy())  # (N,) array
        # area_weights = areas / areas.mean()  # normalize around 1.0
        # num_splats_per_triangle = np.clip((area_weights * num_splats).astype(int), 0, 4)
        
        # Uniform splatting density
        num_splats_per_triangle = np.full(triangles.shape[0], 1, dtype=int)
        print("num_splats_per_triangle shape:", num_splats_per_triangle.shape)
        print("Max and min:", num_splats_per_triangle.max(), num_splats_per_triangle.min())
        
        filter_path = Path("/mnt/data1/samk/NEU/dataset/hotdog/num_of_gaussians.npy")
        if filter_path.exists(): 
            print("Loading splat filter from:", filter_path)
            loaded_filter = np.load(filter_path)
            print("loaded_filter shape:", loaded_filter.shape)
            print("Max and min:", loaded_filter.max(), loaded_filter.min())
            num_splats_per_triangle = num_splats_per_triangle * loaded_filter
            print("Adjusted num_splats_per_triangle shape:", num_splats_per_triangle.shape)
        
        print("Max and min:", num_splats_per_triangle.max(), num_splats_per_triangle.min())
        
        # Since this data set has no colmap data, we start with random points
        num_pts = num_splats_per_triangle.sum()
        print(
            f"Generating random point cloud ({num_pts})..."
        )
        
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
        
        # We create random points inside the bounds traingles
        xyz_list = []
        alpha_list = []
        color_list = []
        
        for i in range(triangles.shape[0]):
            n = num_splats_per_triangle[i]
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
        
        tri_indices = []
        for i in range(triangles.shape[0]):
            n = num_splats_per_triangle[i]
            tri_indices.append(torch.full((n,), i, dtype=torch.long))
        
        tri_indices = torch.cat(tri_indices, dim=0)
        
        
        
        # # >>>> [YC] Prepare UV
        # face_uvs = mesh_scene.visual.uv[faces]  # (n_faces, 3, 2)
        # H, W = texture_img.shape[:2]        
        
        # # Convert UVs (3 per face) to pixel coordinates
        # px = (face_uvs[..., 0] * (W - 1)).astype(int)
        # py = ((1 - face_uvs[..., 1]) * (H - 1)).astype(int)
        
        # # Clamp
        # px = np.clip(px, 0, W - 1)
        # py = np.clip(py, 0, H - 1)
        
        # # Sample 3 vertex colors per triangle
        # tri_vertex_colors = texture_img[py, px, :3]   # (n_faces, 3, 3)
        
        # # Average per-triangle color
        # tri_avg_colors = tri_vertex_colors.mean(axis=1)  # (n_faces, 3)
        # # <<<< [YC] 
        
        # # We create random points inside the bounds traingles
        # alpha = torch.rand(
        #     triangles.shape[0],
        #     num_splats, #! [YC] note: this part decide how many points per triangle
        #     3 
        # )
        # xyz = torch.matmul(
        #     alpha,
        #     triangles
        # )
        # xyz = xyz.reshape(num_splats * triangles.shape[0], 3)
        # print(alpha.shape, xyz.shape) # [YC] debug
        
        # # Repeat each triangle’s color for its num_pts_each_triangle points
        # colors = np.repeat(tri_avg_colors, num_splats, axis=0)  # (num_pts, 3)
        
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
