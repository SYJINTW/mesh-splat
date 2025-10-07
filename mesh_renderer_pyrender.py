import os
import numpy as np
import torch
import trimesh
import pyrender
import imageio

# Load scene (some .obj or .stl files load as a scene with multiple parts)
# mesh_or_scene = trimesh.load("/home/syjintw/Desktop/NEU/dataset/test_mesh/plate_Y-Z.obj")
mesh_or_scene = trimesh.load("/home/syjintw/Desktop/NEU/dataset/test_mesh/hotdog_Y_Z.obj")

# If it's a Scene, extract geometries
if isinstance(mesh_or_scene, trimesh.Scene):
    # Merge all geometries in scene into a single Trimesh
    mesh_trimesh = mesh_or_scene.dump().sum()
else:
    mesh_trimesh = mesh_or_scene

# Convert to pyrender.Mesh
mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)
# mesh.visual.vertex_colors = [180, 180, 180, 255]

width = 800
height = 800
# fx = width / (2 * 1111.111)
# fy = height / (2 * 1111.111)
fx = 1111.111
fy = 1111.111
cx = width / 2
cy = height / 2

cam_pose = np.eye(4)
# cam_pose[:3, 3] = [0, 0, 5]
cam_pose[:3, 3] = [0.0, 0, 2.9592916965484624]
cam_pose[:3, :3] = np.array([
    [-1.0, 0.0, 0.0],
    [ 0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0]
])

# # Camera data from 3DGS
# position = np.array([0.0, 2.7372601032257076, 2.9592916965484624])
# rotation = np.array([
#     [-1.0, -0.0, -0.0],
#     [ 0.0,  0.73411, -0.67903],
#     [-0.0, -0.67903, -0.73411]
# ])

# # 3DGS: camera looks along -Z → negate Z axis to get +Z (view direction for PyRedner)
# forward = -rotation[:, 2]  # invert Z
# look_at = position + forward

# # Y axis of camera frame is the up vector
# up = rotation[:, 1]


camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=1000)


scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0], ambient_light=[3.0, 3.0, 3.0])
scene.add(mesh)

# Add camera node with pose
scene.add(camera, pose=cam_pose)

# Add directional light
light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
scene.add(light, pose=cam_pose)  # Place it at same camera position

# Render offscreen
r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
color, depth = r.render(scene)

# Save image
from PIL import Image
Image.fromarray(color).save("../rendered_pyrender.png")