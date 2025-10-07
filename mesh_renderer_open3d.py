import numpy as np
import open3d as o3d

# Input data
fx = 1111.1110311937682
fy = 1111.1110311937682
width = 800
height = 800
cx = width / 2
cy = height / 2

# Intrinsic matrix for Open3D

intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
# Rotation matrix
R = np.array([
    [-1.0, -0.0, -0.0],
    [ 0.0,  0.73411, -0.67903],
    [-0.0, -0.67903, -0.73411]
])

# Position (camera origin in world coordinates)
position = np.array([0.0, 2.73726, 2.95929])

# Construct 4x4 camera-to-world matrix
c2w = np.eye(4)
c2w[:3, :3] = R
c2w[:3, 3] = position

# Open3D needs world-to-camera
extrinsic = np.linalg.inv(c2w)

# Load mesh
mesh = o3d.io.read_triangle_mesh("/home/syjintw/Desktop/NEU/dataset/test_mesh/hotdog.obj")
mesh.compute_vertex_normals()


vis = o3d.visualization.Visualizer()
vis.create_window(width=width, height=height)
vis.add_geometry(mesh)

ctr = vis.get_view_control()
params = o3d.camera.PinholeCameraParameters()
params.intrinsic = intrinsic
params.extrinsic = extrinsic
ctr.convert_from_pinhole_camera_parameters(params)

vis.run()
vis.destroy_window()
