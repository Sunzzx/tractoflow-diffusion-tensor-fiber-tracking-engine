"""
TractoFlow Interactive Demo
============================
End-to-end pipeline:
  1. Generate synthetic curved + straight fiber phantom DWI
  2. Fit DTI tensors + FA/MD maps
  3. Seed-based RK4 streamline tractography
  4. Connectome extraction + topology metrics
  5. Terminal ANSI visualization of FA map and streamlines
"""

from __future__ import annotations
import sys
import time
import numpy as np

from tractoflow.phantom import create_curved_bundle_phantom
from tractoflow.dti import fit_dti_volume
from tractoflow.tractography import track_ensemble
from tractoflow.connectome import build_parcellation_atlas, extract_connectome


# ANSI color helpers
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BLUE = "\033[94m"
DIM = "\033[2m"


def fa_to_ansi(fa_val: float) -> str:
    """Map FA [0,1] to ANSI color block."""
    if fa_val < 0.1:
        return "\033[48;5;232m  "  # near black
    elif fa_val < 0.25:
        return "\033[48;5;17m  "   # deep blue
    elif fa_val < 0.4:
        return "\033[48;5;27m  "   # blue
    elif fa_val < 0.55:
        return "\033[48;5;39m  "   # cyan
    elif fa_val < 0.7:
        return "\033[48;5;46m  "   # green
    elif fa_val < 0.85:
        return "\033[48;5;226m  "  # yellow
    else:
        return "\033[48;5;196m  "  # red (high FA)


def render_fa_slice(fa: np.ndarray, z: int, title: str = "FA Map") -> str:
    """Render a 2D FA axial slice as colored terminal blocks."""
    slice_2d = fa[:, :, z]
    Nx, Ny = slice_2d.shape
    lines = [f"{BOLD}{CYAN}── {title} (z={z}) ──{RESET}"]
    # Downsample if too large for terminal
    step_x = max(1, Nx // 40)
    step_y = max(1, Ny // 40)
    for ix in range(0, Nx, step_x):
        row = ""
        for iy in range(0, Ny, step_y):
            row += fa_to_ansi(float(slice_2d[ix, iy]))
        lines.append(row + RESET)
    lines.append(
        f"{DIM}FA legend: black(0) → blue → cyan → green → yellow → red(1.0){RESET}"
    )
    return "\n".join(lines)


def render_streamlines_ascii(
    streamlines,
    grid_size: tuple[int, int, int],
    z_plane: int,
    title: str = "Streamline Projections"
) -> str:
    """Project streamlines onto a 2D ASCII canvas at a given Z plane."""
    Nx, Ny, Nz = grid_size
    canvas_h = min(40, Nx)
    canvas_w = min(60, Ny)
    canvas = [[" " for _ in range(canvas_w)] for _ in range(canvas_h)]

    chars = "·•●◆█"
    for sl_idx, sl in enumerate(streamlines[:80]):
        ch = chars[sl_idx % len(chars)]
        for pt in sl.points:
            x, y, z = pt
            if abs(z - z_plane) > 2.5:
                continue
            cx = int(x / Nx * canvas_h)
            cy = int(y / Ny * canvas_w)
            if 0 <= cx < canvas_h and 0 <= cy < canvas_w:
                canvas[cx][cy] = ch

    lines = [f"{BOLD}{MAGENTA}── {title} (z≈{z_plane}) ──{RESET}"]
    border = "┌" + "─" * canvas_w + "┐"
    lines.append(f"{DIM}{border}{RESET}")
    for row in canvas:
        lines.append(f"{DIM}│{RESET}{''.join(row)}{DIM}│{RESET}")
    lines.append(f"{DIM}└{'─' * canvas_w}┘{RESET}")
    lines.append(f"{DIM}Showing up to 80 streamlines near z={z_plane}{RESET}")
    return "\n".join(lines)


def print_banner() -> None:
    banner = f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════╗
║   TRACTOFLOW — DTI Fiber Tractography & Connectomics Engine     ║
║   Diffusion Tensor Imaging · RK4 Streamlines · Brain Networks   ║
╚══════════════════════════════════════════════════════════════════╝{RESET}
"""
    print(banner)


def main() -> int:
    print_banner()
    t0 = time.perf_counter()

    # ─── Step 1: Phantom ───────────────────────────────────────────
    print(f"{BOLD}{YELLOW}[1/5] Generating synthetic curved-bundle DWI phantom…{RESET}")
    phantom = create_curved_bundle_phantom(
        grid_size=(28, 28, 16),
        snr=40.0,
        b_value=1000.0,
        n_directions=30,
        seed=7,
    )
    Nx, Ny, Nz, K = phantom.dwi.shape
    print(f"      Volume: {Nx}×{Ny}×{Nz}, {K} gradient directions")
    print(f"      Ground-truth fibers: {[f['name'] for f in phantom.ground_truth_fibers]}")
    print(f"      Brain mask voxels: {int(np.sum(phantom.mask))}")

    # ─── Step 2: DTI Fit ───────────────────────────────────────────
    print(f"\n{BOLD}{YELLOW}[2/5] Fitting Diffusion Tensors (OLS Stejskal-Tanner)…{RESET}")
    t_dti = time.perf_counter()
    dti = fit_dti_volume(
        dwi=phantom.dwi,
        bvecs=phantom.bvecs,
        bvals=phantom.bvals,
        mask=phantom.mask,
        clamp_negative_evals=True,
    )
    t_dti_ms = (time.perf_counter() - t_dti) * 1000
    fa_wm = dti.fa[phantom.mask]
    print(f"      Fit time: {t_dti_ms:.1f} ms")
    print(f"      Mean FA (masked): {float(np.mean(fa_wm)):.4f}")
    print(f"      Mean MD (masked): {float(np.mean(dti.md[phantom.mask])):.6f} mm²/s")
    print(f"      FA range: [{float(np.min(fa_wm)):.3f}, {float(np.max(fa_wm)):.3f}]")

    # ─── Step 3: Seed generation ───────────────────────────────────
    print(f"\n{BOLD}{YELLOW}[3/5] Generating white-matter seeds (FA > 0.3)…{RESET}")
    seed_coords = np.argwhere((dti.fa > 0.3) & phantom.mask)
    # Subsample uniformly
    rng = np.random.default_rng(42)
    n_seeds = min(120, len(seed_coords))
    if len(seed_coords) > n_seeds:
        idx = rng.choice(len(seed_coords), size=n_seeds, replace=False)
        seed_coords = seed_coords[idx]
    seeds = [tuple(float(c) for c in s) for s in seed_coords]
    print(f"      Seeds selected: {len(seeds)}")

    # ─── Step 4: Tractography ──────────────────────────────────────
    print(f"\n{BOLD}{YELLOW}[4/5] RK4 bidirectional streamline tracking…{RESET}")
    # Principal eigenvector field
    v1 = dti.eigenvectors[..., :, 0]  # (Nx, Ny, Nz, 3)

    t_track = time.perf_counter()
    tractogram = track_ensemble(
        v1_field=v1,
        fa_map=dti.fa,
        seeds=seeds,
        step_size=0.5,
        fa_threshold=0.15,
        max_angle_deg=40.0,
        min_length=4.0,
        max_steps=300,
        method="rk4",
        n_probabilistic_samples=1,
        seed=42,
    )
    t_track_ms = (time.perf_counter() - t_track) * 1000
    print(f"      Tracking time: {t_track_ms:.1f} ms")
    print(f"      Streamlines recovered: {tractogram.count}")
    print(f"      Mean fiber length: {tractogram.mean_length:.2f} voxels")
    if tractogram.count > 0:
        lengths = [sl.length for sl in tractogram.streamlines]
        print(f"      Length range: [{min(lengths):.1f}, {max(lengths):.1f}]")

    # ─── Step 5: Connectome ────────────────────────────────────────
    print(f"\n{BOLD}{YELLOW}[5/5] Building structural connectome…{RESET}")
    atlas = build_parcellation_atlas(grid_size=(Nx, Ny, Nz), n_regions=6)
    graph = extract_connectome(tractogram, atlas, n_regions=6)
    print(f"      Network density: {graph.density:.3f}")
    print(f"      Clustering coefficient: {graph.clustering_coefficient:.3f}")
    print(f"      Global efficiency: {graph.global_efficiency:.4f}")
    print(f"      Node degrees: {graph.node_degrees.astype(int).tolist()}")
    print(f"      Node strengths: {[round(s, 1) for s in graph.node_strengths.tolist()]}")

    # Adjacency preview
    print(f"\n{BOLD}Adjacency Matrix (fiber counts):{RESET}")
    print("     " + "  ".join(f"R{i+1:d}" for i in range(6)))
    for i in range(6):
        row = "  ".join(f"{int(graph.adjacency[i, j]):3d}" for j in range(6))
        print(f"  R{i+1}  {row}")

    # ─── Visualizations ────────────────────────────────────────────
    print()
    mid_z = Nz // 2
    print(render_fa_slice(dti.fa, z=mid_z, title="Fractional Anisotropy (FA)"))
    print()
    print(render_streamlines_ascii(
        tractogram.streamlines,
        grid_size=(Nx, Ny, Nz),
        z_plane=mid_z,
        title="RK4 Streamline Projections"
    ))

    total_ms = (time.perf_counter() - t0) * 1000
    print(f"""
{BOLD}{GREEN}✓ Pipeline complete in {total_ms:.0f} ms{RESET}
{DIM}Humanity Impact: Non-invasive mapping of white-matter pathways enables
early detection of neurodegeneration (MS, Alzheimer's, TBI), surgical
planning for tumor resection near critical tracts, and personalized
connectome-based psychiatric biomarkers.{RESET}
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
