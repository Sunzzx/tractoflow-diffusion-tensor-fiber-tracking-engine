"""
Synthetic Diffusion MRI Phantom Generator.
Generates realistic crossed fiber bundles, curved pathways, and isotropic background noise
with Stejskal-Tanner gradient schemes.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class SyntheticDWIDataset:
    dwi: NDArray[np.float64]        # (Nx, Ny, Nz, K)
    bvecs: NDArray[np.float64]      # (K, 3)
    bvals: NDArray[np.float64]      # (K,)
    ground_truth_fibers: list[dict] # description of fiber geometries
    mask: NDArray[np.bool_]         # (Nx, Ny, Nz)


def generate_gradient_scheme(n_directions: int = 32, b_value: float = 1000.0, n_b0: int = 4) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generate electrostatic repulsion / Fibonacci sphere uniform gradient directions.
    """
    # Baseline b0 volumes
    bvecs = [np.array([0.0, 0.0, 0.0]) for _ in range(n_b0)]
    bvals = [0.0 for _ in range(n_b0)]

    # Golden spiral distribution on sphere
    indices = np.arange(0, n_directions, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_directions)
    theta = np.pi * (1 + 5**0.5) * indices

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    for i in range(n_directions):
        vec = np.array([x[i], y[i], z[i]])
        vec = vec / np.linalg.norm(vec)
        bvecs.append(vec)
        bvals.append(b_value)

    return np.array(bvecs, dtype=np.float64), np.array(bvals, dtype=np.float64)


def create_curved_bundle_phantom(
    grid_size: tuple[int, int, int] = (32, 32, 20),
    snr: float = 30.0,
    b_value: float = 1000.0,
    n_directions: int = 30,
    seed: int = 42
) -> SyntheticDWIDataset:
    """
    Create a 3D synthetic phantom with two distinct white matter tracts:
    1. Tract A (Horizontal / Curved C-loop in XY plane): U-fiber from left to right.
    2. Tract B (Straight Vertical tract along Z axis): corticospinal-like bundle.
    """
    rng = np.random.default_rng(seed)
    Nx, Ny, Nz = grid_size
    bvecs, bvals = generate_gradient_scheme(n_directions=n_directions, b_value=b_value, n_b0=4)
    K = len(bvals)

    dwi = np.zeros((Nx, Ny, Nz, K), dtype=np.float64)
    mask = np.zeros((Nx, Ny, Nz), dtype=bool)

    # Baseline signal S0
    s0_val = 1000.0

    # Diffusivity parameters in mm^2/s (typical brain tissue: ~ 1.5e-3 parallel, 0.3e-3 perpendicular)
    d_par = 1.7e-3
    d_perp = 0.3e-3
    d_iso = 0.8e-3  # isotropic CSF / gray matter background

    # Coordinate grids centered
    x = np.arange(Nx) - Nx / 2.0
    y = np.arange(Ny) - Ny / 2.0
    z = np.arange(Nz) - Nz / 2.0
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Bundle 1: Straight vertical fiber column along Z axis at X ~ [-4, 4], Y ~ [-4, 4]
    dist_z_bundle = np.sqrt(X**2 + Y**2)
    mask_bundle_z = dist_z_bundle <= 4.0

    # Bundle 2: Curved semicircular bundle in XY plane around (0, 0, Z_mid)
    # Radius R = 8.0, cross-section radius r = 3.0
    r_xy = np.sqrt(X**2 + Y**2)
    dist_ring = np.abs(r_xy - 9.0)
    mask_curved = (dist_ring <= 3.0) & (np.abs(Z) <= 4.0) & (X >= -2.0)

    # Brain sphere mask
    dist_center = np.sqrt(X**2 + Y**2 + Z**2)
    mask = dist_center <= (min(Nx, Ny, Nz) * 0.45)

    # Synthesize Stejskal-Tanner DWI signal for each voxel
    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                if not mask[ix, iy, iz]:
                    # Background: air (very low S0)
                    dwi[ix, iy, iz, :] = 10.0
                    continue

                # Determine principal fiber orientation v
                is_z = mask_bundle_z[ix, iy, iz]
                is_curve = mask_curved[ix, iy, iz]

                if is_z and not is_curve:
                    # Fiber along Z: [0, 0, 1]
                    v = np.array([0.0, 0.0, 1.0])
                    # D = d_par * (v @ v^T) + d_perp * (I - v @ v^T)
                    D = d_par * np.outer(v, v) + d_perp * (np.eye(3) - np.outer(v, v))
                elif is_curve and not is_z:
                    # Tangent vector to circle in XY plane: (-Y, X, 0)
                    tan_x = -Y[ix, iy, iz]
                    tan_y = X[ix, iy, iz]
                    norm_tan = np.sqrt(tan_x**2 + tan_y**2) + 1e-9
                    v = np.array([tan_x / norm_tan, tan_y / norm_tan, 0.0])
                    D = d_par * np.outer(v, v) + d_perp * (np.eye(3) - np.outer(v, v))
                elif is_z and is_curve:
                    # Crossing region: average tensor representation
                    v1 = np.array([0.0, 0.0, 1.0])
                    tan_x = -Y[ix, iy, iz]
                    tan_y = X[ix, iy, iz]
                    norm_tan = np.sqrt(tan_x**2 + tan_y**2) + 1e-9
                    v2 = np.array([tan_x / norm_tan, tan_y / norm_tan, 0.0])
                    D1 = d_par * np.outer(v1, v1) + d_perp * (np.eye(3) - np.outer(v1, v1))
                    D2 = d_par * np.outer(v2, v2) + d_perp * (np.eye(3) - np.outer(v2, v2))
                    D = 0.5 * (D1 + D2)
                else:
                    # Isotropic tissue
                    D = d_iso * np.eye(3)

                # Calculate signal S_k = S0 * exp(-b * g_k^T * D * g_k)
                for k in range(K):
                    b = bvals[k]
                    g = bvecs[k]
                    diff_attenuation = np.exp(-b * (g @ D @ g))
                    dwi[ix, iy, iz, k] = s0_val * diff_attenuation

    # Add Rician noise: S_noisy = sqrt((S + N_r)^2 + N_i^2)
    sigma = s0_val / snr
    noise_r = rng.normal(0, sigma, size=dwi.shape)
    noise_i = rng.normal(0, sigma, size=dwi.shape)
    dwi_noisy = np.sqrt((dwi + noise_r)**2 + noise_i**2)

    info = [
        {"name": "corticospinal_vertical_bundle", "axis": "Z", "voxels": int(np.sum(mask_bundle_z & mask))},
        {"name": "curved_association_u_fiber", "axis": "XY_tangent", "voxels": int(np.sum(mask_curved & mask))}
    ]

    return SyntheticDWIDataset(
        dwi=dwi_noisy,
        bvecs=bvecs,
        bvals=bvals,
        ground_truth_fibers=info,
        mask=mask
    )
