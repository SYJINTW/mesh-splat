

# [DONE] port this module from old codebase to here
# [TODO] implement new policy: texture entropy, YC's error-map/visual-quality-based score
# [TODO] store policy result as .npy file alongside GS/Mesh for reproducibility and batch processing

# ensure all of these run expectedly
# then test out performance
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict

import numpy as np
import torch
import trimesh
from typing import Optional



class BudgetingPolicy(ABC):
    """
    Abstract Base Class of budgeting policies for training.
    """
    def __init__(self, mesh=None):
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
    def __init__(self, mesh=None):
        super().__init__(mesh)

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
    def __init__(self, mesh=None):
        super().__init__(mesh)

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
    Allocates a uniform number of points to each triangle.
    """
    def __init__(self, mesh=None):
        super().__init__(mesh)

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
    def __init__(self, mesh=None):
        super().__init__(mesh)

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


def get_budgeting_policy(name: str, mesh=None) -> BudgetingPolicy:
    REGISTRY: Dict[str, type] = {
        "uniform": UniformBudgetingPolicy,
        "rand_uni": RandomUniformBudgetingPolicy,
        "rand_norm": RandomNormalBudgetingPolicy,
        "area": AreaBasedBudgetingPolicy,
        "planarity": PlanarityBasedBudgetingPolicy,
        "texture": None,
        "mse_mask": None,
        "from_file": None,
    }
    try:
        print(f"[INFO] Budget::Using budgeting policy: {name}")
        return REGISTRY[name](mesh=mesh)
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



def _compute_planarity_mrl(mesh: trimesh.Trimesh, hops: int = 1) -> Optional[np.ndarray]:
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

# NEW: Planarity policy
class PlanarityBasedBudgetingPolicy(BudgetingPolicy):
    """
    Allocate by surface planarity:
    - focus='nonplanar': more splats where neighborhood is non-planar (low MRL).
    - focus='planar': more splats where neighborhood is planar (high MRL).
    """
    # [TODO] try different #hops
    # 1 done
    # 2
    # 3
    
    # [TODO] what about planar focus, will that help? also test to find out
    
    def __init__(self, mesh: Optional[trimesh.Trimesh], hops: int = 1, focus: str = "nonplanar"):
        self.mesh = mesh
        self.hops = int(max(0, hops)) # ensure non-negative
        self.focus = focus.lower()
        self.mrl: Optional[np.ndarray] = None
        if mesh is not None:
            try:
                self.mrl = _compute_planarity_mrl(mesh, hops=self.hops)
            except Exception:
                self.mrl = None

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
    Allocate points based on precomputed distortion/error map per triangle.
    Higher distortion -> more points.
    """
    def __init__(self, mesh: Optional[trimesh.Trimesh], distortion_map: Optional[np.ndarray] = None):
        self.mesh = mesh
        self.distortion_map = distortion_map

    def allocate(
        self,
        triangles: torch.Tensor,
        total_splats: int,
        min_per_tri: int,
        max_per_tri: int
    ) -> np.ndarray:
        
        # [TODO] logic is in train::warmup()
        pass
        # N = int(triangles.shape[0])
        # if self.distortion_map is None or len(self.distortion_map) != N:
        #     # Fallback to uniform if we can't compute distortion map
        #     weights = np.ones((N,), dtype=np.float32)
        # else:
        #     weights = np.maximum(self.distortion_map, 1e-6)  # ensure positive weights
        # return _bounded_proportional_allocate(weights, total_splats, min_per_tri, max_per_tri)



class _TextureGradBasePolicy(BudgetingPolicy):
    """
    texture policies using gradient-energy complexity over UVs.
    """
    
    def __init__(self, mesh: Optional[trimesh.Trimesh], samples_per_tri: int = 16):
        self.mesh = mesh
        self.samples_per_tri = samples_per_tri
        self.complexity: Optional[np.ndarray] = None
        pass
        # if mesh is not None:
        #     try:
        #         self.complexity = _compute_triangle_texture_complexity(
        #             mesh, samples_per_tri=samples_per_tri
        #         )
        #     except Exception:
        #         self.complexity = None

    def allocate(
        self,
        triangles: torch.Tensor,
        total_splats: int,
        min_per_tri: int,
        max_per_tri: int,
        weights: Optional[np.ndarray]
    ) -> np.ndarray:
        N = int(triangles.shape[0])
        if weights is None or len(weights) != N:
            # Fallback to uniform if we can't compute texture complexity
            weights = np.ones((N,), dtype=np.float32)
        return _bounded_proportional_allocate(weights, total_splats, min_per_tri, max_per_tri)
    
    def _try_get_texture_image(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
        """
        Try to fetch a diffuse texture image (H,W,3) in float32 [0,1].
        """
        if mesh is None or mesh.visual is None:
            return None

        # Trimesh texture material path
        try:
            mat = getattr(mesh.visual, "material", None)
            if mat is not None:
                pil_img = getattr(mat, "image", None)
                if pil_img is not None and Image is not None:
                    # Ensure numpy array in [0,1]
                    img = np.asarray(pil_img).astype(np.float32)
                    if img.ndim == 2:
                        img = np.stack([img, img, img], axis=-1)
                    if img.shape[-1] == 4:
                        # Drop alpha
                        img = img[..., :3]
                    if img.max() > 1.0:
                        img = img / 255.0
                    return img
        except Exception:
            pass

        return None

    def _to_grayscale(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img.astype(np.float32)
        # Simple luma
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.astype(np.float32)


    def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
        """
        Approximate gradient magnitude using numpy gradients.
        gray: (H,W) float32
        returns (H,W) float32
        """
        gy, gx = np.gradient(gray)
        mag = np.sqrt(gx * gx + gy * gy)
        # Normalize to [0,1] for stability
        mmax = mag.max()
        if mmax > 0:
            mag = mag / mmax
        return mag.astype(np.float32)


    def _sample_barycentric(n: int) -> np.ndarray:
        """
        Uniform samples over triangle via barycentric coords.
        Returns (n,3) with rows summing to 1.
        """
        r1 = np.random.rand(n).astype(np.float32)
        r2 = np.random.rand(n).astype(np.float32)
        sqrt_r1 = np.sqrt(r1)
        a = 1.0 - sqrt_r1
        b = sqrt_r1 * (1.0 - r2)
        c = sqrt_r1 * r2
        return np.stack([a, b, c], axis=-1)


    def _compute_triangle_texture_complexity(\
        self,
        mesh: trimesh.Trimesh,
        samples_per_tri: int = 16
    ) -> Optional[np.ndarray]:
        """
        Compute per-triangle complexity as mean gradient magnitude under UV sampling.
        Returns (F,) float32 or None if UV/texture unavailable.
        """
        # Need texture image and UVs
        tex = self._try_get_texture_image(mesh)
        uvs = getattr(mesh.visual, "uv", None)
        faces = mesh.faces

        if tex is None or uvs is None or len(uvs) == 0:
            return None

        H, W = tex.shape[0], tex.shape[1]
        gray = _to_grayscale(tex)
        grad_mag = _gradient_magnitude(gray)

        try:
            face_uvs = uvs[faces]  # (F,3,2)
        except Exception:
            # Some meshes have per-face-vertex UVs with distinct indexing; fallback not implemented
            return None

        F = face_uvs.shape[0]
        comp = np.zeros((F,), dtype=np.float32)

        # Vectorized sampling per triangle (loop over faces to keep memory in check)
        for i in range(F):
            uv_tri = face_uvs[i]  # (3,2)
            bary = _sample_barycentric(samples_per_tri)  # (K,3)
            uv = bary @ uv_tri  # (K,2)
            # uv is in [0,1]; map to pixel coords. V likely flipped in images.
            x = np.clip(np.round(uv[:, 0] * (W - 1)).astype(np.int64), 0, W - 1)
            y = np.clip(np.round((1.0 - uv[:, 1]) * (H - 1)).astype(np.int64), 0, H - 1)
            comp[i] = float(grad_mag[y, x].mean())

        # Normalize to [0,1]
        cmax = comp.max()
        if cmax > 0:
            comp = comp / cmax
        return comp

