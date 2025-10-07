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
    print("type(mesh_scene.visual):", type(mesh_scene.visual))
    print("type(mesh_scene.visual):", type(mesh_scene.visual)) # should be TextureVisuals
    print("mesh_scene.visual.uv.shape:", mesh_scene.visual.uv.shape) # UV coordinates per vertex
    print("mesh_scene.visual.material.image:", mesh_scene.visual.material.image) # PIL image
    
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
    # if not os.path.exists(ply_path):
    if True:
        # Since this data set has no colmap data, we start with random points
        num_pts_each_triangle = num_splats
        num_pts = num_pts_each_triangle * triangles.shape[0]
        print(
            f"Generating random point cloud ({num_pts})..."
        )

        # We create random points inside the bounds traingles
        alpha = torch.rand(
            triangles.shape[0],
            num_pts_each_triangle,
            3
        )

        xyz = torch.matmul(
            alpha,
            triangles
        )
        xyz = xyz.reshape(num_pts, 3)

        shs = np.random.random((num_pts, 3)) / 255.0
        
        # >>>> [YC]
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

        # Repeat each triangle’s color for its num_pts_each_triangle points
        colors = np.repeat(tri_avg_colors, num_pts_each_triangle, axis=0)  # (num_pts, 3)
    

        # # Interpolated UVs
        # uvs = np.einsum("fpc,fcv->fpv", alpha.numpy(), face_uvs)  # (n_faces, num_pts_per_face, 2)
        # uvs = uvs.reshape(-1, 2)
        
        # # Clamp to valid range [0,1]
        # uvs = np.clip(uvs, 0.0, 1.0)
        
        # # Convert UVs to pixel coordinates
        # px = (uvs[:, 0] * (W - 1)).astype(int)
        # py = ((1 - uvs[:, 1]) * (H - 1)).astype(int)  # flip Y
        
        # # Extra safety clamp
        # px = np.clip(px, 0, W - 1)
        # py = np.clip(py, 0, H - 1)
        
        # # Sample texture
        # colors = texture_img[py, px, :3]  # (num_pts, 3) in RGB 
        # <<<< [YC]

        pcd = MeshPointCloud(
            alpha=alpha,
            points=xyz,
            # colors=SH2RGB(shs),
            colors=colors/255.0,
            normals=np.zeros((num_pts, 3)),
            vertices=vertices,
            faces=faces,
            transform_vertices_function=transform_vertices_function,
            triangles=triangles.cuda()
        )

        storePly(ply_path, pcd.points, SH2RGB(shs) * 255)

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
