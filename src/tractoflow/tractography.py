"""
Streamline Fiber Tractography Engine.
Includes:
- Deterministic FACT (Fiber Assignment by Continuous Tracking)
- 4th-Order Runge-Kutta (RK4) with trilinear vector field interpolation
- Probabilistic Bootstrapped Direction Tracking with Bingham/von-Mises dispersion
- Stopping criteria (FA threshold, maximum curvature angle, volume bounds)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from numpy.typing import NDArray


@dataclass
class Streamline:
    points: NDArray[np.float64]  # (N_pts, 3) continuous 3D spatial points
    length: float                # total arc length
    fa_profile: NDArray[np.float64] # FA sampled along the trajectory


@dataclass
class Tractogram:
    streamlines: list[Streamline]
    count: int
    mean_length: float


def trilinear_interpolate_vector(
    vector_field: NDArray[np.float64],
    coord: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    """
    Sample a 3D vector field at continuous spatial coordinate (x, y, z) using trilinear interpolation.
    Args:
        vector_field: (Nx, Ny, Nz, 3) 3D array of 3-vectors
        coord: (3,) float array
    Returns:
        interpolated (3,) vector, normalized, or None if outside bounds.
    """
    Nx, Ny, Nz = vector_field.shape[:3]
    x, y, z = coord

    if x < 0 or x >= Nx - 1 or y < 0 or y >= Ny - 1 or z < 0 or z >= Nz - 1:
        return None

    x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    xd = x - x0
    yd = y - y0
    zd = z - z0

    # Interpolate along 8 corners
    # Reference vector for sign disambiguation (since v and -v are identical orientations)
    v000 = vector_field[x0, y0, z0]
    if np.linalg.norm(v000) < 1e-6:
        return None

    def align(v_target: NDArray[np.float64]) -> NDArray[np.float64]:
        if np.dot(v_target, v000) < 0:
            return -v_target
        return v_target

    c000 = v000
    c100 = align(vector_field[x1, y0, z0])
    c010 = align(vector_field[x0, y1, z0])
    c110 = align(vector_field[x1, y1, z0])
    c001 = align(vector_field[x0, y0, z1])
    c101 = align(vector_field[x1, y0, z1])
    c011 = align(vector_field[x0, y1, z1])
    c111 = align(vector_field[x1, y1, z1])

    c00 = c000 * (1 - xd) + c100 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    c = c0 * (1 - zd) + c1 * zd
    norm = np.linalg.norm(c)
    if norm < 1e-6:
        return None
    return c / norm


def trilinear_interpolate_scalar(
    scalar_field: NDArray[np.float64],
    coord: NDArray[np.float64]
) -> float | None:
    """Sample scalar field (e.g. FA map) at continuous coordinate."""
    Nx, Ny, Nz = scalar_field.shape
    x, y, z = coord
    if x < 0 or x >= Nx - 1 or y < 0 or y >= Ny - 1 or z < 0 or z >= Nz - 1:
        return None

    x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    xd, yd, zd = x - x0, y - y0, z - z0

    c00 = scalar_field[x0, y0, z0] * (1 - xd) + scalar_field[x1, y0, z0] * xd
    c10 = scalar_field[x0, y1, z0] * (1 - xd) + scalar_field[x1, y1, z0] * xd
    c01 = scalar_field[x0, y0, z1] * (1 - xd) + scalar_field[x1, y0, z1] * xd
    c11 = scalar_field[x0, y1, z1] * (1 - xd) + scalar_field[x1, y1, z1] * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    return float(c0 * (1 - zd) + c1 * zd)


def rk4_step(
    vector_field: NDArray[np.float64],
    pos: NDArray[np.float64],
    direction: NDArray[np.float64],
    step_size: float
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """
    4th-Order Runge-Kutta numerical integration step with collinear vector alignment:
    k1 = f(y_n)
    k2 = f(y_n + dt/2 * k1)
    k3 = f(y_n + dt/2 * k2)
    k4 = f(y_n + dt * k3)
    y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    def get_aligned_v(p: NDArray[np.float64], ref_dir: NDArray[np.float64]) -> NDArray[np.float64] | None:
        v = trilinear_interpolate_vector(vector_field, p)
        if v is None:
            return None
        if np.dot(v, ref_dir) < 0:
            v = -v
        return v

    k1 = get_aligned_v(pos, direction)
    if k1 is None:
        return None, None

    k2 = get_aligned_v(pos + 0.5 * step_size * k1, k1)
    if k2 is None:
        return None, None

    k3 = get_aligned_v(pos + 0.5 * step_size * k2, k2)
    if k3 is None:
        return None, None

    k4 = get_aligned_v(pos + step_size * k3, k3)
    if k4 is None:
        return None, None

    delta = (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    new_pos = pos + delta
    new_dir = delta / (np.linalg.norm(delta) + 1e-12)
    return new_pos, new_dir


def track_streamline_bidirectional(
    seed: NDArray[np.float64],
    v1_field: NDArray[np.float64],
    fa_map: NDArray[np.float64],
    step_size: float = 0.5,
    fa_threshold: float = 0.15,
    max_angle_deg: float = 45.0,
    max_steps: int = 500,
    method: str = "rk4",
    rng: np.random.Generator | None = None,
    probabilistic_dispersion: float = 0.0
) -> Streamline | None:
    """
    Bidirectional streamline propagation originating from a single seed point.
    """
    initial_fa = trilinear_interpolate_scalar(fa_map, seed)
    if initial_fa is None or initial_fa < fa_threshold:
        return None

    init_v = trilinear_interpolate_vector(v1_field, seed)
    if init_v is None:
        return None

    cos_max_angle = np.cos(np.radians(max_angle_deg))

    # Track forward (+init_v) and backward (-init_v)
    def track_one_branch(start_dir: NDArray[np.float64]) -> list[tuple[NDArray[np.float64], float]]:
        branch = []
        curr_p = seed.copy()
        curr_d = start_dir.copy()

        for _ in range(max_steps):
            if method == "rk4":
                next_p, next_d = rk4_step(v1_field, curr_p, curr_d, step_size)
            else:  # euler / FACT
                v = trilinear_interpolate_vector(v1_field, curr_p)
                if v is None:
                    break
                if np.dot(v, curr_d) < 0:
                    v = -v
                next_p = curr_p + step_size * v
                next_d = v

            if next_p is None or next_d is None:
                break

            # Check probabilistic perturbation if requested
            if probabilistic_dispersion > 0 and rng is not None:
                random_kick = rng.normal(0, probabilistic_dispersion, size=3)
                next_d = next_d + random_kick
                norm_d = np.linalg.norm(next_d)
                if norm_d > 1e-6:
                    next_d = next_d / norm_d

            # Check curvature angle: dot(curr_d, next_d)
            if np.dot(curr_d, next_d) < cos_max_angle:
                break  # Curvature threshold violated

            # Check FA threshold
            fa_val = trilinear_interpolate_scalar(fa_map, next_p)
            if fa_val is None or fa_val < fa_threshold:
                break  # Left white matter or exited mask

            branch.append((next_p, fa_val))
            curr_p = next_p
            curr_d = next_d

        return branch

    fwd = track_one_branch(init_v)
    bwd = track_one_branch(-init_v)

    # Combine: bwd (reversed) + seed + fwd
    bwd_pts = [p for p, fa in reversed(bwd)]
    bwd_fas = [fa for p, fa in reversed(bwd)]

    fwd_pts = [p for p, fa in fwd]
    fwd_fas = [fa for p, fa in fwd]

    all_pts = bwd_pts + [seed] + fwd_pts
    all_fas = bwd_fas + [initial_fa] + fwd_fas

    if len(all_pts) < 3:
        return None

    pts_arr = np.array(all_pts, dtype=np.float64)
    fa_arr = np.array(all_fas, dtype=np.float64)

    # Arc length
    dists = np.linalg.norm(np.diff(pts_arr, axis=0), axis=1)
    length = float(np.sum(dists))

    return Streamline(points=pts_arr, length=length, fa_profile=fa_arr)


def track_ensemble(
    v1_field: NDArray[np.float64],
    fa_map: NDArray[np.float64],
    seeds: Sequence[tuple[float, float, float] | NDArray[np.float64]],
    step_size: float = 0.5,
    fa_threshold: float = 0.15,
    max_angle_deg: float = 45.0,
    min_length: float = 3.0,
    max_steps: int = 500,
    method: str = "rk4",
    n_probabilistic_samples: int = 1,
    probabilistic_dispersion: float = 0.0,
    seed: int = 42
) -> Tractogram:
    """
    Perform deterministic or probabilistic fiber tracking across multiple seeds.
    """
    rng = np.random.default_rng(seed)
    streamlines: list[Streamline] = []

    for s in seeds:
        s_arr = np.array(s, dtype=np.float64)
        for _ in range(n_probabilistic_samples):
            sl = track_streamline_bidirectional(
                seed=s_arr,
                v1_field=v1_field,
                fa_map=fa_map,
                step_size=step_size,
                fa_threshold=fa_threshold,
                max_angle_deg=max_angle_deg,
                max_steps=max_steps,
                method=method,
                rng=rng if probabilistic_dispersion > 0 else None,
                probabilistic_dispersion=probabilistic_dispersion
            )
            if sl is not None and sl.length >= min_length:
                streamlines.append(sl)

    count = len(streamlines)
    mean_len = float(np.mean([sl.length for sl in streamlines])) if count > 0 else 0.0
    return Tractogram(streamlines=streamlines, count=count, mean_length=mean_len)
