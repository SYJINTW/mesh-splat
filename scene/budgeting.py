from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, Optional, Tuple 
from types import SimpleNamespace
from functools import partial
from tqdm import tqdm

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

import torchvision.transforms as T
import torchvision.transforms.functional as TF



EPS = 1e-8 # small positive epsilon to avoid divide-by-zero

# ensure all of the policies run expectedly
# then test out performance of each one of them
class BudgetingPolicy(ABC):
    """
    Abstract Base Class of budgeting policies for training.
    Subclasses should compute `self.weights` in their __init__ method.
    """
    def __init__(self, mesh: trimesh.Trimesh, **kwargs):  # accept and ignore extra keyword arguments
        # what if mesh is in other formats, e.g. pytorch3d meshes?
        assert mesh is not None, "BudgetingPolicy requires a mesh object."
        self.mesh = mesh
        self.num_triangles = int(mesh.faces.shape[0])
        # Default to uniform weights if not overridden by a subclass
        self.weights: np.ndarray = np.ones((self.num_triangles,), dtype=np.float32)

    def allocate(
        self,
        total_splats: int
        ) -> np.ndarray:
        """
        Default allocate method without bounds.
        returns a list (numpy array) of number of splats per triangle
        """
        
        # Check if all weights are equal (uniform distribution)
        # note, np handles array-like for us
        if np.allclose(self.weights, self.weights[0], rtol=1e-5):
            print(f"[DEBUG] All weights are equal ({self.weights[0]:.6f}). "
                  "Allocation will be uniform. "
                  "If policy != uniform, weight computation might have failed."
                  )
        
        return _unbounded_proportional_allocate(self.weights, total_splats)


    def allocate_bounded(
        self,
        total_splats: int,
        min_per_tri: int,
        max_per_tri: int
    ) -> np.ndarray:                    # shape [N], dtype=int
        """
        returns a list (numpy array) of number of splats per triangle
        """
        return _bounded_proportional_allocate(
            self.weights, total_splats, min_per_tri, max_per_tri
        )
        
    def drop(self): # or keep(self)
        # placeholder for future ABR algorithms
        # could be used in render/post-processing
        pass


class MixedBudgetingPolicy(BudgetingPolicy):
    # 2D loss (distortion) map + 3D geometric property (e.g. mesh geometry)
    weight_distortion: float = 0.5 # hyperparameter to be tuned empirically
    weight_geometry: float = 1.0 - weight_distortion
    pass


class AreaBasedBudgetingPolicy(BudgetingPolicy):
    """
    Allocates points to triangles based on their surface area.
    Larger triangles get more points.
    """
    def __init__(self, mesh: trimesh.Trimesh, **kwargs):
        super().__init__(mesh, **kwargs)
        # Use pre-computed face areas from the trimesh object
        self.weights = np.maximum(self.mesh.area_faces, EPS).astype(np.float32)


class UniformBudgetingPolicy(BudgetingPolicy):
    """
    Allocates a uniform number of points to each triangle.
    """
    def __init__(self, mesh=None, **kwargs):
        super().__init__(mesh, **kwargs)
    
    # only this baseline policy overrides allocate() method
    def allocate(
        self,
        total_splats: int,
    ) -> np.ndarray:
        num_triangles = self.num_triangles
        uniform_alloc = total_splats // num_triangles
        print(f"[INFO] Budget::UniformBudgetingPolicy allocating {uniform_alloc} splats per triangle.")
        
        return np.full((num_triangles,), uniform_alloc, dtype=np.int32)

class RandomUniformBudgetingPolicy(BudgetingPolicy):
    """
    Allocates points to triangles based on weights randomly sampled from Uniform(0,1).
    """
    def __init__(self, mesh: trimesh.Trimesh, **kwargs):
        super().__init__(mesh, **kwargs)
        weights = np.random.rand(self.num_triangles).astype(np.float32)
        self.weights = np.clip(weights, EPS, 1.0) # make sure weights are positive



class RandomNormalBudgetingPolicy(BudgetingPolicy):
    """
        Allocates points to triangles based on weights randomly sampled from Normal(0,1).

    """
    def __init__(self, mesh: trimesh.Trimesh, **kwargs):
        super().__init__(mesh, **kwargs)
        mu = 0.5
        sigma = 0.15 # adjustable parameter
        w = np.random.normal(loc=mu, scale=sigma, size=self.num_triangles).astype(np.float32)
        self.weights = np.clip(w, EPS, 1.0) # make sure weights are positive

    

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
        
        # same, use partial to set focus or other params
        "texture": None, 
        "texture_focus": None,
        "texture_avoid": None,
        
        
        "distortion": DistortionMapBudgetingPolicy,
        
        "distortion_no_avg": partial(DistortionMapBudgetingPolicy, is_averaging_across_views=False),
        
        "from_file": None, # currently handled in dataset_reader::get_num_splats_per_triangle
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

    ###########################################################################
    # [NOTE] the [min,max] part could be ignored if we're just using [0, inf) # 
    ###########################################################################
    
    # 3. Iteratively distribute the remaining budget
    # Normalize weights to prevent very large numbers, ensure they are positive
    w_sum = np.sum(weights)
    if w_sum > 0:
        norm_weights = weights / w_sum
    else:
        # If all weights are zero, fallback to uniform weights
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
    
    # Final check to ensure budget is fully exhausted
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
    assert np.all(alloc >= min_per) and np.all(alloc <= max_per), "Allocation violates min/max bounds"

    return alloc


#[DONE] try unbounded, namely, [0, inf)
def _unbounded_proportional_allocate(
    weights: np.ndarray,
    total: int,
    # min_per: int,
    # max_per: int
) -> np.ndarray:
    """
    input: weight/importance/priority/score per triangle

    Allocate nonnegative *integers* that:
    - sum exactly to 'total'
    - proportional to 'weights' (when possible)
    
    Returns:
        np.ndarray: Array of integers of shape (N,) representing the allocation per triangle.
    
    ---
    Example:
    weights = [0.1, 0.2, 0.4], total=10
    initial allocation = [1.428, 2.857, 5.714], 
    integer part = [1, 2, 5], remainder = [0.428, 0.857, 0.714]
    final allocation = [1, 3, 6]

    """

    N = weights.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=np.int32)
    
    # Normalize weights to prevent very large numbers, ensure they are positive
    w_sum = np.sum(weights)
    if w_sum > 0:
        norm_weights = weights / w_sum
        # [TODO] could try other normalization strategies
        # e.g. exponential, logarithmic, etc.
    else:
        # If all weights are zero, fallback to uniform weights
        norm_weights = np.ones(N, dtype=np.float32) / N
        print("[WARNING] sum of all weights are zero; distributing uniformly.")

    print(f"[DEBUG] Weights stats - min: {weights.min():.4f}, max: {weights.max():.4f}, mean: {weights.mean():.4f}, stdv: {weights.std():.4f}")
    



    # "Largest Remainder Method"
    alloc =  np.zeros((N,), dtype=np.int32)
    
    
    # Keep track of fractional parts to decide who gets the next splat
    fractional_parts = norm_weights * total
    
    # Distribute the integer part of the proportional allocation
    int_alloc = fractional_parts.astype(np.int32)
    alloc += int_alloc
    
    # the remainder one by one based on largest fractional part
    budget_to_distribute = total - alloc.sum()
    remainder = fractional_parts - int_alloc
    indices_to_add = np.argsort(-remainder) # Sort descending
    for i in range(budget_to_distribute):
        idx = indices_to_add[i % N] # Cycle through if needed, though unlikely
        alloc[idx] += 1
    
    
    # calculate the correlation coefficient 
    # to see how far off alloc[]:int is from weights[]:float
    expected_alloc = norm_weights * total  # ideal fractional allocation
    if np.allclose(norm_weights, norm_weights[0], rtol=1e-9):
        correlation = 1.0  # Perfect correlation for uniform distribution
    else:
        correlation = np.corrcoef(norm_weights, alloc / total)[0,1]
        # the result is [[1, corr],[corr,1]]
    rmse = np.sqrt(np.mean((alloc - expected_alloc) ** 2))
    
    print(f"[DEBUG] Allocation quality metrics:")
    print(f"  - Pearson correlation: {correlation:.4f} (1.0 = perfect)")
    print(f"  - RMSE: {rmse:.4f} (0.0 = perfect)")


    assert np.all(alloc >= 0), "Error: Allocation contains negative values"
    assert alloc.sum() == total, f"Error: Final allocation sum {alloc.sum()} does not match total budget {total}"
    return alloc



class PlanarityBasedBudgetingPolicy(BudgetingPolicy):
    """
    Allocate by surface planarity:
    - focus='nonplanar': more splats where neighborhood is non-planar (low MRL).
    - focus='planar': more splats where neighborhood is planar (high MRL).
    """
    
    def __init__(self, mesh: trimesh.Trimesh, hops: int = 1, focus: str = "nonplanar", **kwargs):
        super().__init__(mesh, **kwargs)
        self.hops = int(max(0, hops))
        self.focus = focus.lower()
        try:
            mrl = self._compute_planarity_mrl(mesh, hops=self.hops)
            if mrl is not None and len(mrl) == self.num_triangles:
                if self.focus == "planar":
                    self.weights = np.maximum(mrl, EPS)  # high on planar
                    
                else: # 'nonplanar'
                    # [TODO] could change 1-F_i to other types of inversion functions, e.g. exp(-k*F_i)
                    self.weights = np.maximum(1.0 - mrl, EPS)  # high on non-planar
                
                print(f"[INFO] Budget::PlanarityBasedBudgetingPolicy using focus='{self.focus}' with hops={self.hops}")
            else:
                print(f"[WARNING] PlanarityBasedBudgetingPolicy: Failed to compute planarity, falling back to uniform weights.")

        except Exception as e:
            print(f"[WARNING] PlanarityBasedBudgetingPolicy: Failed to compute planarity, falling back to uniform weights. Error: {e}")
            # Fallback to uniform weights is handled by the base class __init__


    def _compute_planarity_mrl(self, mesh: trimesh.Trimesh, hops: int = 1) -> Optional[np.ndarray]:
        """
        Returns per-face mean resultant length (MRL) in [0,1], 1 = planar neighborhood.
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

        # [DOING] [TODO] [BUG] Investigate adjacency list construction for meshes with non-standard topology.
        # [WORKAROUND] give the few neighbor-less faces 0 weight or something
        # There may be issues if the mesh is not watertight or has missing/incorrect face adjacency.
        # Build adjacency list from face_adjacency
        adj = [[] for _ in range(F)]
        
        # Check if mesh is watertight/has valid topology
        if not mesh.is_watertight:
            print(f"[WARNING] Mesh is not watertight - may have incomplete adjacency")
        
        # Access face_adjacency - this triggers computation
        fa = mesh.face_adjacency
        
        if fa is None or len(fa) == 0:
            print(f"[WARNING] PlanarityBasedBudgetingPolicy: Mesh has no face adjacency.")
            print(f"[DEBUG] Mesh has {len(mesh.edges)} edges, {len(mesh.edges_unique)} unique edges")
            print(f"[DEBUG] This usually means the mesh has no shared edges between triangles")
            return np.ones((F,), dtype=np.float32)
        
        print(f"[DEBUG] Found {len(fa)} face adjacency pairs for {F} faces")
        
        # Build adjacency list
        for a, b in fa:
            adj[int(a)].append(int(b))
            adj[int(b)].append(int(a))
        
        # Verify adjacency was built
        num_isolated = sum(1 for neighbors in adj if len(neighbors) == 0)
        if num_isolated > 0:
            print(f"[WARNING] {num_isolated}/{F} faces have no neighbors")

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
            
            m = float(np.clip(m, 0.0, 1.0)) # clamp to [0,1]
            mrl[i] = m

        return mrl




#[FIXED] the policy was not correctly loaded during render_mesh_splat
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
        mesh: trimesh.Trimesh,
        viewpoint_camera_infos=None,  # pass in CamInfo, get Camera later
        dataset_path: str = None,
        faces_per_pixel: int = 1,
        device: str = "cuda",
        debugging: bool = True,
        p3d_mesh: Meshes = None, 
        is_averaging_across_views: bool = True,
        **kwargs
    ):
        super().__init__(mesh, **kwargs)
        self.viewpoint_camera_infos = viewpoint_camera_infos
        self.dataset_path = dataset_path
        self.faces_per_pixel = faces_per_pixel
        self.device = device
        self.debugging = debugging
        self.p3d_mesh = p3d_mesh  # Store the passed-in mesh
        self.is_averaging_across_views = is_averaging_across_views
        if self.is_averaging_across_views:
            print(f"[INFO] DistortionMapBudgeter:: Averaging distortion across views")
        else: 
            print(f"[INFO] DistortionMapBudgeter:: Not averaging distortion across views")
        

        assert self.viewpoint_camera_infos is not None and len(self.viewpoint_camera_infos) != 0, "DistorsionMapPolicy::Missing CamInfos"

        # Build Camera objects
        args = SimpleNamespace(resolution= -1, data_device=device) # dummy args
        
        # this camera should be freed after use?
        self.viewpoint_cameras = cameraList_from_camInfos(
            self.viewpoint_camera_infos, resolution_scale=1.0, 
            args=args
        )
        assert isinstance(self.viewpoint_cameras[0], Camera), "DistorsionMapPolicy::can't get Camera objects for view_points"

        # Compute distortion weights and assign to self.weights
        distortion_weights = self._compute_distortion_weights()
        if distortion_weights is not None and len(distortion_weights) == self.num_triangles:
            self.weights = distortion_weights
            print(f"[INFO] DistortionMapBudgetingPolicy: Using computed distortion weights")
            print(f"[INFO] Weight stats - min: {self.weights.min():.4f}, max: {self.weights.max():.4f}, mean: {self.weights.mean():.4f}")
        else:
            print("[WARNING] DistortionMapBudgetingPolicy: No valid distortion weights, falling back to uniform")
            # Fallback to uniform is handled by base class __init__

    def _load_or_create_mesh(self) -> Tuple[Meshes, torch.Tensor, torch.Tensor]:  # Use Tuple instead of tuple
        """
        Helper method to load textured mesh or create from trimesh.
        
        Returns:
            tuple: (p3d_mesh, verts, faces) where:
                - p3d_mesh: PyTorch3D Meshes object with textures
                - verts: Vertex tensor [V, 3]
                - faces: Face tensor [F, 3]
        """
        # If p3d_mesh was provided in __init__, use it directly
        if self.p3d_mesh is not None:
            print("[DEBUG] Using provided p3d_mesh")
            
            assert isinstance(self.p3d_mesh, Meshes), "[ERROR] Provided p3d_mesh is not a PyTorch3D Meshes object"
            return (
                self.p3d_mesh,
                self.p3d_mesh.verts_packed(),
                self.p3d_mesh.faces_packed()
            )
        
        # Otherwise, load from file or create from trimesh
        if self.dataset_path:
            mesh_path = f"{self.dataset_path}/mesh.obj"
            if os.path.exists(mesh_path):
                p3d_mesh = load_objs_as_meshes([mesh_path]).to(self.device)
                print(f"[DEBUG] Loaded textured mesh from {mesh_path} using PyTorch3D load_objs_as_meshes()")
                
                # Extract vertices and faces WITH PyTorch3D's coordinate system transforms
                verts = p3d_mesh.verts_packed()
                faces = p3d_mesh.faces_packed()
                return p3d_mesh, verts, faces
        
        # Fallback: create from trimesh
        print("[DEBUG] Creating mesh from Trimesh (no textured mesh found)")
        verts = torch.tensor(self.mesh.vertices, dtype=torch.float32, device=self.device)
        faces = torch.tensor(self.mesh.faces, dtype=torch.int64, device=self.device)
        
        # Create white textured mesh
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb)
        p3d_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
        
        return p3d_mesh, verts, faces

    # [DONE] check the heatmap point cloud against the mesh, the coordinates should align
    def _compute_distortion_weights(self) -> np.ndarray:
        """
        Compute per-triangle distortion weights by rendering from all viewpoints.
        Uses batched rendering for efficiency.
        """
        if self.mesh is None or self.viewpoint_cameras is None:
            print("[WARNING] DistortionMapBudgetingPolicy: Missing mesh or cameras")
            return None
        
        print("[INFO] DistortionMapBudgetingPolicy:: Computing distortion weights...")
        
        # Load or use provided mesh
        p3d_mesh, verts, faces = self._load_or_create_mesh()

        # Create mesh for rasterization (white texture for face indexing)
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb)
        tm2p3d_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)

        num_faces = faces.shape[0]
        dist_map_all = torch.zeros(num_faces, dtype=torch.float32, device=self.device)
        per_view_debug = [] if self.debugging else None
        
        
        # [TODO] this is not really batch processing, it's still sequential
        # Batch processing with tqdm
        batch_size = 8  # Process 8 cameras at once - adjust based on GPU memory
        num_cameras = len(self.viewpoint_cameras)
        num_batches = (num_cameras + batch_size - 1) // batch_size
        
        print(f"[INFO] Processing {num_cameras} cameras in {num_batches} batches of {batch_size}")
        
        # Create progress bar for batches
        batch_pbar = tqdm(
            range(0, num_cameras, batch_size),
            desc="Distortion:: Processing camera batches",
            total=num_batches,
            unit="batch",
            ncols=100
        )
        
        for batch_start in batch_pbar:
            batch_end = min(batch_start + batch_size, num_cameras)
            batch_cameras = self.viewpoint_cameras[batch_start:batch_end]
            actual_batch_size = len(batch_cameras)
            
            # Update progress bar description
            batch_pbar.set_description(
                f"Batch {batch_start//batch_size + 1}/{num_batches} "
                f"(cams {batch_start}-{batch_end-1})"
            )
            
            # Process each camera in the batch
            for local_idx, viewpoint_camera in enumerate(batch_cameras):
                idx = batch_start + local_idx
                
                # Get camera-specific dimensions
                cam_height = viewpoint_camera.image_height
                cam_width = viewpoint_camera.image_width
                
                # Get ground truth image - already [C, H, W] on GPU
                gt_img = viewpoint_camera.original_image  # [C, H, W]
                
                # Render textured mesh
                p3d_mesh_color_rgb, _, _ = mesh_renderer_pytorch3d(
                    viewpoint_camera, p3d_mesh,
                    image_height=cam_height,
                    image_width=cam_width,
                    faces_per_pixel=self.faces_per_pixel,
                    device=self.device
                )
                # the rendering function doesn't support batching yet
                
                p3d_mesh_color_rgb = torch.clamp(p3d_mesh_color_rgb, 0.0, 1.0)
                
                # Compute per-pixel absolute difference - [C, H, W] format
                dist_map = torch.mean(torch.abs(gt_img - p3d_mesh_color_rgb), dim=0)  # [H, W]
                
                # Render face indices
                _, _, tm2p3d_fragments = mesh_renderer_pytorch3d(
                    viewpoint_camera, tm2p3d_mesh,
                    image_height=cam_height,
                    image_width=cam_width,
                    faces_per_pixel=self.faces_per_pixel,
                    device=self.device
                )
                
                # Pixel-to-face mapping
                face_idx_map = tm2p3d_fragments.pix_to_face[0, ..., 0]  # [H, W]
                
                # Flatten and filter
                face_idx_flat = face_idx_map.flatten()
                dist_flat = dist_map.flatten()
                valid_mask = face_idx_flat >= 0
                face_idx_flat = face_idx_flat[valid_mask]
                dist_flat = dist_flat[valid_mask]
                
                # Accumulate using bincount
                sum_dist = torch.bincount(face_idx_flat, weights=dist_flat, minlength=num_faces)
                count = torch.bincount(face_idx_flat, minlength=num_faces).float()
                
                mean_dist = torch.zeros(num_faces, dtype=torch.float32, device=self.device)
                mask = count > 0
                
                if self.is_averaging_across_views:
                    
                    mean_dist[mask] = sum_dist[mask] / count[mask]
                else:
                    mean_dist[mask] = sum_dist[mask]
                    
                
                # Accumulate distortion
                dist_map_all += mean_dist
                
                # Debug info
                if per_view_debug is not None:
                    per_view_debug.append({
                        "index": idx,
                        "image_name": getattr(viewpoint_camera, "image_name", f"view_{idx}"),
                        "p3d_mesh_color": p3d_mesh_color_rgb.cpu(),
                        "gt": gt_img.cpu(),
                        "dist_map": dist_map.cpu().numpy(),
                    })
        
            # Free memory after each batch
            if batch_end < num_cameras:
                torch.cuda.empty_cache()
    
        batch_pbar.close()
        
        # Move final result to CPU
        dist_map_all_np = dist_map_all.cpu().numpy()
        
        if self.debugging:
            print(f"[DEBUG] Distortion weights (pre-normalization) stats - min: {dist_map_all_np.min():.4f}, max: {dist_map_all_np.max():.4f}, mean: {dist_map_all_np.mean():.4f}")
            self._save_debug_visualization(dist_map_all_np, per_view_debug=per_view_debug)
            
            # check mesh format and coordinate systems
            print(f"[DEBUG] p3d_mesh verts range: {p3d_mesh.verts_packed().min():.4f} to {p3d_mesh.verts_packed().max():.4f}")
            print(f"[DEBUG] tm2p3d_mesh verts range: {tm2p3d_mesh.verts_packed().min():.4f} to {tm2p3d_mesh.verts_packed().max():.4f}")
            print(f"[DEBUG] Verts match: {torch.allclose(p3d_mesh.verts_packed(), tm2p3d_mesh.verts_packed())}")
            print(f"[DEBUG] Faces match: {torch.equal(p3d_mesh.faces_packed(), tm2p3d_mesh.faces_packed())}")
            
            dist_norm = (dist_map_all_np - dist_map_all_np.min()) / (dist_map_all_np.max() - dist_map_all_np.min() + EPS)
            print(f"[DEBUG] Distortion weights computed:")
            print(f"  - Non-zero triangles: {np.count_nonzero(dist_norm)}/{num_faces}")
            print(f"  - Weight range: [{dist_norm.min():.4f}, {dist_norm.max():.4f}]")
        else:
            dist_norm = (dist_map_all_np - dist_map_all_np.min()) / (dist_map_all_np.max() - dist_map_all_np.min() + EPS)
    
        assert dist_map_all.max() >= 0, "Distortion map contains negative values."
        
        return np.maximum(dist_norm, EPS).astype(np.float32)

    def _save_debug_visualization(self, dist_map_all: np.ndarray, per_view_debug=None):
        """Save distortion map debug artifacts: per-view renders, heatmaps, and colored point cloud."""
        try:
            import matplotlib.cm as cm
            import matplotlib.pyplot as plt
            import open3d as o3d

            base_dir = "./distortion_debug_visualization"
            heatmap_dir = os.path.join(base_dir, "heatmap")
            mesh_bg_dir = os.path.join(base_dir, "mesh_bg")
            os.makedirs(heatmap_dir, exist_ok=True)
            os.makedirs(mesh_bg_dir, exist_ok=True)

            # 1) Per-view artifacts
            if per_view_debug is not None:
                for item in per_view_debug:
                    name = str(item.get("image_name", f"view_{item.get('index', 0)}"))
                    p3d_mesh_color = item["p3d_mesh_color"]
                    # print(f"[DEBUG] shape: {p3d_mesh_color.shape}, type: {type(p3d_mesh_color)}")
                    # Save rendered mesh background
                    try:
                        p3d_mesh_color_pil = TF.to_pil_image(p3d_mesh_color.cpu())
                        p3d_mesh_color_pil.save(os.path.join(mesh_bg_dir, f"{name}.png"))
                    except Exception as e:
                        print(f"[WARNING] Could not save render for {name}: {e} (got {getattr(item['render'], 'shape', None)})")
                        pass

                    # Save heatmap (normalized)
                    try:
                        dm = item["dist_map"].astype(np.float32)
                        dm_norm = dm / (np.max(dm) + EPS)
                        plt.figure(figsize=(6, 6))
                        plt.imshow(dm_norm, cmap='hot', vmin=0.0, vmax=1.0)
                        plt.axis('off')
                        plt.tight_layout(pad=0)
                        plt.savefig(os.path.join(heatmap_dir, f"{name}.png"), dpi=200, bbox_inches='tight', pad_inches=0)
                        plt.close()
                    except Exception as e:
                        print(f"[WARNING] Could not save heatmap for {name}: {e}")

            # 2) Global PLY with per-vertex colors aggregated from per-face distortion
            dist_norm = (dist_map_all - dist_map_all.min()) / (dist_map_all.ptp() + EPS)
            cmap = cm.get_cmap('jet')
            colors = cmap(dist_norm)[:, :3]  # (num_faces, 3), RGB in [0,1]
            
            # Compute per-vertex color by averaging colors of adjacent faces
            vertex_colors = np.zeros((len(self.mesh.vertices), 3))
            for f_id, verts in enumerate(self.mesh.faces):
                vertex_colors[verts] += colors[f_id]
            counts = np.bincount(self.mesh.faces.flatten(), minlength=len(self.mesh.vertices))
            vertex_colors /= np.maximum(counts[:, None], EPS)

            # Build Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.mesh.vertices)
            pcd.colors = o3d.utility.Vector3dVector(vertex_colors)

            # Save as PLY
            output_path = os.path.join(base_dir, "distortion_heatmap.ply")
            o3d.io.write_point_cloud(output_path, pcd)
            print(f"[INFO] Saved distortion debug to {base_dir}")
        except Exception as e:
            print(f"[WARNING] Could not save debug visualization: {e}")







