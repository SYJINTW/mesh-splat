import os
import numpy as np
import torch
import trimesh
import pyrender
import imageio

# Load scene (some .obj or .stl files load as a scene with multiple parts)
mesh_or_scene = trimesh.load("/home/syjintw/Desktop/NEU/dataset/nerf_synthetic/hotdog/mesh.obj")

# If it's a Scene, extract geometries
if isinstance(mesh_or_scene, trimesh.Scene):
    # Merge all geometries in scene into a single Trimesh
    mesh_trimesh = mesh_or_scene.dump().sum()
else:
    mesh_trimesh = mesh_or_scene

# Convert to pyrender.Mesh
mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)

fx = 1111.111
fy = 1111.111
width = 800
height = 800
cx = width / 2
cy = height / 2

camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=100)

# Rotation matrix (3x3) and translation vector
R = np.array([
    [-1.0, -0.0, -0.0],
    [ 0.0,  0.73411, -0.67903],
    [-0.0, -0.67903, -0.73411]
])
t = np.array([0.0, 2.73726, 2.95929])

# 4x4 camera-to-world pose
c2w = np.eye(4)
c2w[:3, :3] = R
c2w[:3, 3] = t

# Convert to world-to-camera
w2c = np.linalg.inv(c2w)

scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0], ambient_light=[1.0, 1.0, 1.0])
scene.add(mesh)

# Add camera node with pose
scene.add(camera, pose=w2c)

# Add directional light
light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
scene.add(light, pose=w2c)  # Place it at same camera position

# Render offscreen
r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
color, depth = r.render(scene)

# Save image
from PIL import Image
Image.fromarray(color).save("rendered_pyrender.png")