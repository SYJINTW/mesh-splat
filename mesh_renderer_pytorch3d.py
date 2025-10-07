import torch
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesVertex,
)
import imageio
import numpy as np

# 使用 CUDA 如果可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 建立球體 Mesh
mesh = ico_sphere(level=1, device=device)

# 建立每個 vertex 的顏色為白色 (R=1, G=1, B=1)
verts_rgb = torch.ones_like(mesh.verts_padded())  # (B, V, 3)
textures = TexturesVertex(verts_features=verts_rgb.to(device))
mesh.textures = textures

# 相機、光源、光柵化設定
cameras = FoVPerspectiveCameras(device=device)
raster_settings = RasterizationSettings(image_size=256)
lights = PointLights(device=device)

# 建立渲染器
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
)

# 執行渲染
images = renderer(mesh)

# 顯示 shape
print("Rendered image shape:", images.shape)  # torch.Size([1, 256, 256, 3])

# 取得 RGB (前3個通道)
image = images[0, ..., :3].cpu().numpy()  # (256, 256, 3)

# 選擇是否需要考慮透明度：通常不需要，直接取 RGB 即可
image = (image * 255).astype(np.uint8)
imageio.imwrite("rendered_sphere.png", image)