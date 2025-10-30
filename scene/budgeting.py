# [DONE] port this module from old codebase to here
# [TODO] store policy result as .npy file alongside GS/Mesh for reproducibility and batch processing

# ensure all of these run expectedly
# then test out performance
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, Optional
from functools import partial

import numpy as np
import torch
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.io import load_objs_as_meshes
from mesh_renderer_pytorch3d import mesh_renderer_pytorch3d
import cv2
import os 
import matplotlib.cm as cm
from scene.cameras import Camera

from utils.camera_utils import cameraList_from_camInfos




class BudgetingPolicy(ABC):
    """
    Abstract Base Class of budgeting policies for training.
    """
    def __init__(self, mesh=None, **kwargs):  # accept and ignore extra keyword arguments
        # some policies might not need mesh
        self.mesh = mesh

    @abstractmethod
    def allocate(
        self,
        triangles: torch.Tensor,        # [N,3,3]
        total_splats: int,
        min_per_tri: int,
        max_per_tri: int
    ) -> np.ndarray:                    # shape [N], dtype=int
        """
        returns a list (numpy array) of number of splats per triangle
        """
        pass

    def drop(self):
        # placeholder for future
        # could be used in render/post-processing
        pass



class AreaBasedBudgetingPolicy(BudgetingPolicy):
    """
    Allocates points to triangles based on their surface area.
    Larger triangles get more points.
    """
    def __init__(self, mesh=None, **kwargs):
        super().__init__(mesh, **kwargs)

    def allocate(
        self,
        triangles: torch.Tensor,
        total_splats: int,
        min_per_tri: int,
        max_per_tri: int
    ) -> np.ndarray:
        areas = trimesh.triangles.area(triangles.cpu().numpy())
        # Use area as weights for the allocation function
        return _bounded_proportional_allocate(areas, total_splats, min_per_tri, max_per_tri)



class UniformBudgetingPolicy(BudgetingPolicy):
    """
    Allocates a uniform number of points to each triangle.
    """
    def __init__(self, mesh=None, **kwargs):
        super().__init__(mesh, **kwargs)

    def allocate(
        self,
        triangles: torch.Tensor,
        total_splats: int,
        min_per_tri: int,
        max_per_tri: int
    ) -> np.ndarray:
        num_triangles = triangles.shape[0]
        # For uniform allocation, all weights are equal
        weights = np.ones(num_triangles, dtype=np.float32)
        return _bounded_proportional_allocate(weights, total_splats, min_per_tri, max_per_tri)


class RandomUniformBudgetingPolicy(BudgetingPolicy):
    """
    Allocates points to triangles based on weights randomly sampled from Uniform(0,1).
    """
    def __init__(self, mesh=None, **kwargs):
        super().__init__(mesh, **kwargs)

    def allocate(
        self,
        triangles: torch.Tensor,
        total_splats: int,
        min_per_tri: int,
        max_per_tri: int
    ) -> np.ndarray:
        num_triangles = triangles.shape[0]
        weights = np.random.rand(num_triangles).astype(np.float32)
        weights = np.clip(weights, 1e-6, 1.0) # make sure weights are positive
        return _bounded_proportional_allocate(weights, total_splats, min_per_tri, max_per_tri)



class RandomNormalBudgetingPolicy(BudgetingPolicy):
    """
    Allocates a uniform number of points to each triangle.
    """
    def __init__(self, mesh=None, **kwargs):
        super().__init__(mesh, **kwargs)

    def allocate(
        self,
        triangles: torch.Tensor,
        total_splats: int,
        min_per_tri: int,
        max_per_tri: int
    ) -> np.ndarray:
        num_triangles = triangles.shape[0]
        mu = 0.5
        sigma = 0.15 # adjustable parameter
        w = np.random.normal(loc=mu, scale=sigma, size=num_triangles).astype(np.float32)
        weights = np.clip(w, 1e-6, 1.0) # make sure weights are positive
        return _bounded_proportional_allocate(weights, total_splats, min_per_tri, max_per_tri)


def get_budgeting_policy(name: str, mesh=None, **kwargs) -> BudgetingPolicy:
    
    REGISTRY: Dict[str, type] = {
        "uniform": UniformBudgetingPolicy, 
        "random": RandomUniformBudgetingPolicy, # turns out to be better than naive Uniform
        # "rand_norm": RandomNormalBudgetingPolicy,
        "area": AreaBasedBudgetingPolicy,
        
        # [TODO] try different #hops, then change this to the optimal candidate
        "planarity": partial(PlanarityBasedBudgetingPolicy, hops=1), 
        
        # expose more interfaces instead of hardcoding and manually testing
        "planarity1": partial(PlanarityBasedBudgetingPolicy, hops=1),
        "planarity2": partial(PlanarityBasedBudgetingPolicy, hops=2),
        "planarity3": partial(PlanarityBasedBudgetingPolicy, hops=3),
        
        # same, use partial to set focus
        "texture": None, 
        "texture_focus": None,
        "texture_avoid": None,
        
        
        "distortion": DistortionMapBudgetingPolicy,
        "from_file": None,
    }
    try:
        print(f"[INFO] Budget::Using budgeting policy: {name}")
        policy_class = REGISTRY[name]
        if policy_class is None:
            raise NotImplementedError(f"Policy '{name}' is not yet implemented")
        return policy_class(mesh=mesh, **kwargs)
    except KeyError:
        raise ValueError(f"Unknown budgeting policy: '{name}'")


def _bounded_proportional_allocate(
    weights: np.ndarray,
    total: int,
    min_per: int,
    max_per: int
) -> np.ndarray:
    """
    input: weight/importance/priority/score per triangle
    
    Allocate integers that:
    - sum exactly to 'total'
    - each in [min_per, max_per]
    - proportional to 'weights' (when possible)
    """
    N = weights.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=np.int32)
    
    max_possible = N * max_per
    if total > max_possible:
        print(f"[WARNING] Requested budget {total} exceeds the maximum possible #={max_possible}. "
              f"Capping at {max_per} splats per triangle, for a new total of {max_possible}.")
        return np.full(N, max_per, dtype=np.int32)

    min_required = N * min_per
    if total < min_required:
        # For the lower bound, raising an error is usually better as it's an unrecoverable state.
        raise ValueError(f"Total budget {total} is less than the minimum required {min_required}")

    # 1. Start with the minimum allocation for everyone
    alloc = np.full(N, min_per, dtype=np.int32)
    
    # 2. Calculate remaining budget to distribute
    remaining_budget = total - alloc.sum()
    assert remaining_budget >= 0, "Remaining budget should be non-negative"
    if remaining_budget == 0:
        return alloc

    # 3. Iteratively distribute the remaining budget
    # Normalize weights to prevent very large numbers, ensure they are positive
    w_sum = np.sum(weights)
    if w_sum > 0:
        norm_weights = weights / w_sum
    else:
        # If all weights are zero, distribute uniformly
        norm_weights = np.ones(N, dtype=np.float32) / N
        print("[WARNING] sum of all weights are zero; distributing uniformly.")

    # Keep track of fractional parts to decide who gets the next splat
    # "Largest Remainder Method"
    fractional_parts = norm_weights * remaining_budget
    
    # Distribute the integer part of the proportional allocation
    int_alloc = fractional_parts.astype(np.int32)
    
    # Check capacity constraints
    capacity = max_per - min_per
    int_alloc = np.minimum(int_alloc, capacity)
    
    alloc += int_alloc
    
    # 4. Distribute the final remainder one by one based on largest fractional part
    budget_to_distribute = total - alloc.sum()
    remainder = fractional_parts - int_alloc
    
    # Use sorting to give splats to those with the largest remainder
    indices_to_add = np.argsort(-remainder) # Sort descending

    for i in range(budget_to_distribute):
        idx = indices_to_add[i % N] # Cycle through if needed, though unlikely
        if alloc[idx] < max_per:
            alloc[idx] += 1
    
    # Final check to ensure budget is met exactly
    final_sum = alloc.sum()
    if final_sum != total:
        # If there's still a discrepancy (due to max_per cap), adjust greedily
        deficit = total - final_sum
        if deficit > 0:
            for idx in indices_to_add:
                if deficit == 0: break
                can_add = max_per - alloc[idx]
                add_amount = min(deficit, can_add)
                alloc[idx] += add_amount
                deficit -= add_amount
        elif deficit < 0:
            for idx in reversed(indices_to_add):
                if deficit == 0: break
                can_remove = alloc[idx] - min_per
                remove_amount = min(-deficit, can_remove)
                alloc[idx] -= remove_amount
                deficit += remove_amount

    assert alloc.sum() == total, f"Final allocation sum {alloc.sum()} does not match total budget {total}"
    assert np.all(alloc >= min_per) and np.all(alloc <= max_per), "Allocation is out of min/max bounds"

    return alloc



class PlanarityBasedBudgetingPolicy(BudgetingPolicy):
    """
    Allocate by surface planarity:
    - focus='nonplanar': more splats where neighborhood is non-planar (low MRL).
    - focus='planar': more splats where neighborhood is planar (high MRL).
    """
    
    def __init__(self, mesh: Optional[trimesh.Trimesh], hops: int = 1, focus: str = "nonplanar", **kwargs):
        super().__init__(mesh, **kwargs)
        self.hops = int(max(0, hops))
        self.focus = focus.lower()
        self.mrl: Optional[np.ndarray] = None
        if mesh is not None:
            try:
                self.mrl = self._compute_planarity_mrl(mesh, hops=self.hops)
            except Exception:
                self.mrl = None

    def _compute_planarity_mrl(self, mesh: trimesh.Trimesh, hops: int = 1) -> Optional[np.ndarray]:
        """
        Returns per-face mean resultant length in [0,1], 1 = planar neighborhood.
        Uses face adjacency up to 'hops'.
        """
        if mesh is None or mesh.faces is None or mesh.face_normals is None:
            return None
        F = int(mesh.faces.shape[0])
        if F == 0:
            return None

        normals = mesh.face_normals.astype(np.float32)
        # Normalize to unit vectors (defensive)
        n_norm = np.linalg.norm(normals, axis=1, keepdims=True)
        n_norm[n_norm == 0] = 1.0
        normals = normals / n_norm

        adj = [[] for _ in range(F)]
        fa = getattr(mesh, "face_adjacency", None)
        if fa is None or fa.size == 0:
        # No adjacency info; fall back to per-face normal magnitude (all 1)
            return np.ones((F,), dtype=np.float32)
        for a, b in fa:
            adj[int(a)].append(int(b))
            adj[int(b)].append(int(a))

        def neighborhood(seed: int) -> np.ndarray:
            if hops <= 0:
                return np.array([seed], dtype=np.int64)
            visited = {seed}
            q = deque([(seed, 0)])
            while q:
                v, d = q.popleft()
                if d == hops:
                    continue
                for u in adj[v]:
                    if u not in visited:
                        visited.add(u)
                        q.append((u, d + 1))
            return np.fromiter(visited, dtype=np.int64)

        mrl = np.zeros((F,), dtype=np.float32)
        for i in range(F):
            nb = neighborhood(i)
            mean_n = normals[nb].mean(axis=0)
            m = np.linalg.norm(mean_n).astype(np.float32)
        # clamp to [0,1]
            m = float(np.clip(m, 0.0, 1.0))
            mrl[i] = m

        return mrl

    def allocate(
        self,
        triangles: torch.Tensor,
        total_splats: int,
        min_per_tri: int,
        max_per_tri: int
    ) -> np.ndarray:
        N = int(triangles.shape[0])
        if self.mrl is None or len(self.mrl) != N:
            weights = np.ones((N,), dtype=np.float32)
        else:
            if self.focus == "planar":
                weights = np.maximum(self.mrl, 1e-6)  # high on planar
            else:
                # [TODO] could change 1-F_i to other types of inversion functions, e.g. exp(-k*F_i)
                weights = np.maximum(1.0 - self.mrl, 1e-6)  # high on non-planar
                
            print(f"[INFO] Budget::PlanarityBasedBudgetingPolicy using focus='{self.focus}' with hops={self.hops}")
            
            
        return _bounded_proportional_allocate(weights, total_splats, min_per_tri, max_per_tri)





class DistortionMapBudgetingPolicy(BudgetingPolicy):
    """
    Allocate points based on distortion/error of rendering textured mesh vs ground truths.
    Higher distortion -> more points.
    
    Computes per-triangle distortion by:
    1. Rendering mesh from each viewpoint
    2. Computing per-pixel error vs ground truth
    3. Mapping pixel errors to triangles via rasterization
    4. Accumulating mean error per triangle across views
    """
    def __init__(
        self, 
        mesh: Optional[trimesh.Trimesh],
        viewpoint_camera_infos=None,  # pass in CamInfo, get Camera later
        dataset_path: str = None,
        image_height: int = 800,
        image_width: int = 800,
        faces_per_pixel: int = 1,
        device: str = "cuda",
        debugging: bool = True,
        **kwargs
    ):
        super().__init__(mesh, **kwargs)
        self.viewpoint_camera_infos = viewpoint_camera_infos
        self.dataset_path = dataset_path
        self.image_height = image_height
        self.image_width = image_width
        self.faces_per_pixel = faces_per_pixel
        self.device = device
        self.debugging = debugging
        self.distortion_weights: Optional[np.ndarray] = None

        assert mesh is not None, "DistorsionMapPolicy::Missing Mesh "
        assert self.viewpoint_camera_infos is not None and len(self.viewpoint_camera_infos) != 0, "DistorsionMapPolicy::Missing CamInfos"

        # Build Camera objects
        self.viewpoint_cameras = cameraList_from_camInfos(
            self.viewpoint_camera_infos, resolution_scale=1.0, 
            args={"resolution": 1, "data_device": device}
        )
        assert isinstance(self.viewpoint_cameras[0], Camera), "DistorsionMapPolicy::can't get Camera objects for view_points"

        # Compute distortion weights
        self.distortion_weights = self._compute_distortion_weights()

    def _load_with_white_bg(self, path):
        """Load image with white background compositing."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")

        if len(img.shape) == 2:
            # Grayscale
            img_out = img.astype(np.float32) / 255.0
            img_out = np.stack([img_out] * 3, axis=-1)
        elif img.shape[2] == 4:  # RGBA
            rgb = img[:, :, :3].astype(np.float32) / 255.0
            alpha = img[:, :, 3:].astype(np.float32) / 255.0  # shape [H,W,1]
            white_bg = np.ones_like(rgb)
            img_out = rgb * alpha + white_bg * (1 - alpha)
        else:
            img_out = img[:, :, :3].astype(np.float32) / 255.0

        # Convert BGR to RGB because OpenCV loads in BGR order
        img_out = img_out[:, :, ::-1]
        return img_out

    def _compute_distortion_weights(self) -> np.ndarray:
        """
        Compute per-triangle distortion weights by rendering from all viewpoints.
        """
        if self.mesh is None or self.viewpoint_cameras is None:
            print("[WARNING] DistortionMapBudgetingPolicy: Missing mesh or cameras")
            return None
        
        print("[INFO] DistortionMapBudgetingPolicy:: Computing distortion weights...")
        
        # Convert trimesh to PyTorch3D meshes
        verts = torch.tensor(self.mesh.vertices, dtype=torch.float32, device=self.device)
        faces = torch.tensor(self.mesh.faces, dtype=torch.int64, device=self.device)
        
        # Create mesh for rasterization (white texture for face indexing)
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb)
        tm2p3d_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
        
        # Load textured mesh for rendering
        if self.dataset_path:
            mesh_path = f"{self.dataset_path}/mesh.obj"
            if os.path.exists(mesh_path):
                p3d_mesh = load_objs_as_meshes([mesh_path]).to(self.device)
            else:
                print(f"[WARNING] Textured mesh not found at {mesh_path}, using white Trimesh mesh")
                p3d_mesh = tm2p3d_mesh
        else:
            print("[WARNING] No dataset_path provided, using white mesh")
            p3d_mesh = tm2p3d_mesh
        
        num_faces = faces.shape[0]
        dist_map_all = np.zeros(num_faces, dtype=np.float32)
        
        for idx, viewpoint_camera in enumerate(self.viewpoint_cameras):
            # Load ground truth image
            gt_image_path = f"{self.dataset_path}/mesh_texture/{viewpoint_camera.image_name}.png"
                
            if not os.path.exists(gt_image_path):
                print(f"[WARNING] Ground truth image not found: {gt_image_path}")
                continue
                
            gt_img = self._load_with_white_bg(gt_image_path)
            
            # Render textured mesh
            p3d_mesh_color, _, _ = mesh_renderer_pytorch3d(
                viewpoint_camera, p3d_mesh,
                image_height=self.image_height,
                image_width=self.image_width,
                faces_per_pixel=self.faces_per_pixel,
                device=self.device
            )
            
            # Convert to numpy
            p3d_mesh_color_np = (
                p3d_mesh_color[0, ..., :3]
                .detach().cpu().numpy()
            )
            p3d_mesh_color_np = np.clip(p3d_mesh_color_np, 0.0, 1.0).astype(np.float32)
            
            # Compute per-pixel absolute difference map
            dist_map = np.mean(np.abs(gt_img - p3d_mesh_color_np), axis=2)  # [H, W]
            
            # Render face indices
            _, _, tm2p3d_fragments = mesh_renderer_pytorch3d(
                viewpoint_camera, tm2p3d_mesh,
                image_height=self.image_height,
                image_width=self.image_width,
                faces_per_pixel=self.faces_per_pixel,
                device=self.device
            )
            
            # Face index per pixel
            face_idx_map = tm2p3d_fragments.pix_to_face[0, ..., 0].cpu().numpy()  # [H, W]
            
            # Flatten arrays
            face_idx_flat = face_idx_map.flatten()
            dist_flat = dist_map.flatten()
            
            # Filter out invalid faces (background = -1)
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
            
            # Accumulate distortion across views
            dist_map_all += mean_dist
        
        if self.debugging:
            self._save_debug_visualization(dist_map_all)
        
        # Normalize to [0, 1] and ensure positive weights
        assert dist_map_all.max() >= 0, "Distortion map contains negative values."
        dist_norm = (dist_map_all - dist_map_all.min()) / (dist_map_all.max() - dist_map_all.min() + 1e-8)
        
        return np.maximum(dist_norm, 1e-6).astype(np.float32)

    def _save_debug_visualization(self, dist_map_all: np.ndarray):
        """Save distortion map as colored point cloud for visualization."""
        try:
            import matplotlib.cm as cm
            import open3d as o3d
            
            # Normalize for colormap
            dist_norm = (dist_map_all - dist_map_all.min()) / (dist_map_all.ptp() + 1e-8)
            cmap = cm.get_cmap('jet')
            colors = cmap(dist_norm)[:, :3]  # (num_faces, 3), RGB in [0,1]
            
            # Compute per-vertex color by averaging colors of adjacent faces
            vertex_colors = np.zeros((len(self.mesh.vertices), 3))
            for f_id, verts in enumerate(self.mesh.faces):
                vertex_colors[verts] += colors[f_id]
                
            counts = np.bincount(self.mesh.faces.flatten(), minlength=len(self.mesh.vertices))
            vertex_colors /= np.maximum(counts[:, None], 1e-8)

            # Build Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.mesh.vertices)
            pcd.colors = o3d.utility.Vector3dVector(vertex_colors)

            # Save as PLY
            output_path = "./debug_distortion_heatmap.ply"
            o3d.io.write_point_cloud(output_path, pcd)
            print(f"[INFO] Saved distortion heatmap to {output_path}")
        except Exception as e:
            print(f"[WARNING] Could not save debug visualization: {e}")

    def allocate(
        self,
        triangles: torch.Tensor,
        total_splats: int,
        min_per_tri: int,
        max_per_tri: int
    ) -> np.ndarray:
        N = int(triangles.shape[0])
        
        if self.distortion_weights is None or len(self.distortion_weights) != N:
            # Fallback to uniform if we can't compute distortion
            weights = np.ones((N,), dtype=np.float32)
            print("[WARNING] DistortionMapBudgetingPolicy: No valid distortion weights, falling back to uniform")
        else:
            weights = self.distortion_weights
            print(f"[INFO] DistortionMapBudgetingPolicy: Using computed distortion weights")
            print(f"[INFO] Weight stats - min: {weights.min():.4f}, max: {weights.max():.4f}, mean: {weights.mean():.4f}")
        
        return _bounded_proportional_allocate(weights, total_splats, min_per_tri, max_per_tri)

