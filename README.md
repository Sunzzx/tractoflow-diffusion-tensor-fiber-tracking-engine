# TractoFlow

### Diffusion Tensor Imaging (DTI) & Deterministic / Probabilistic Streamline Fiber Tractography Engine for Brain Connectomics

[![CI](https://github.com/Sunzzx/tractoflow-diffusion-tensor-fiber-tracking-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Sunzzx/tractoflow-diffusion-tensor-fiber-tracking-engine/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?logo=github)](https://github.com/sponsors/Sunzzx)
[![Polar](https://img.shields.io/badge/Support-Polar.sh-blue)](https://polar.sh/Sunzzx)

> **TractoFlow** reconstructs three-dimensional white-matter fiber pathways from multi-shell Diffusion-Weighted MRI (dMRI) using classical Diffusion Tensor Imaging, 4th-order Runge–Kutta streamline integration, and graph-theoretic structural connectomics — enabling non-invasive mapping of the human brain's wiring diagram.

---

## Humanity Impact

White-matter tracts are the high-bandwidth communication highways of the brain. Damage or malformation of these pathways underlies:

- **Multiple Sclerosis & demyelinating disease** — FA collapse is an early biomarker before clinical disability
- **Traumatic Brain Injury (TBI) & concussion** — diffuse axonal injury invisible on standard T1/T2 MRI becomes quantifiable via tractography
- **Alzheimer's & frontotemporal dementia** — progressive disconnection of association fibers tracks cognitive decline
- **Neurosurgical planning** — tumor resection near the corticospinal tract, arcuate fasciculus, or optic radiations requires millimeter-accurate pathway maps to preserve motor, language, and vision
- **Psychiatric biomarkers** — altered structural connectome topology (efficiency, clustering) correlates with schizophrenia, depression, and autism spectrum conditions
- **Pediatric neurodevelopment** — tracking myelination trajectories from infancy through adolescence

TractoFlow provides a pure-Python, dependency-light research engine for generating synthetic ground-truth phantoms, fitting DTI tensors, tracking fibers, and extracting connectomes — suitable for algorithm development, teaching, and offline clinical research pipelines.

---

## Theoretical Foundations

### 1. Stejskal–Tanner Signal Model

Diffusion-weighted MRI attenuates the baseline signal $S_0$ according to the local diffusion tensor $\mathbf{D} \in \mathbb{R}^{3 \times 3}$ (symmetric, positive semi-definite):

$$
S_k = S_0 \exp\!\bigl(-b_k\,\mathbf{g}_k^{\mathsf{T}} \mathbf{D}\,\mathbf{g}_k\bigr)
$$

where $b_k$ is the $b$-value (s/mm²) and $\mathbf{g}_k$ is the unit gradient direction for acquisition $k$.

Taking the logarithm yields a linear system in the six unique tensor components:

$$
\ln S_k = \ln S_0 - b_k \bigl(
  g_x^2 D_{xx} + 2 g_x g_y D_{xy} + 2 g_x g_z D_{xz}
  + g_y^2 D_{yy} + 2 g_y g_z D_{yz} + g_z^2 D_{zz}
\bigr)
$$

In matrix form: $\mathbf{y} = \mathbf{B}\,\boldsymbol{\beta}$, solved via ordinary least squares

$$
\hat{\boldsymbol{\beta}} = \mathbf{B}^{+}\,\mathbf{y}, \qquad \mathbf{B}^{+} = (\mathbf{B}^{\mathsf{T}}\mathbf{B})^{-1}\mathbf{B}^{\mathsf{T}}
$$

### 2. Tensor Eigendecomposition & Scalar Invariants

The fitted tensor is diagonalized:

$$
\mathbf{D} = \mathbf{V}\,\boldsymbol{\Lambda}\,\mathbf{V}^{\mathsf{T}}, \quad
\boldsymbol{\Lambda} = \mathrm{diag}(\lambda_1, \lambda_2, \lambda_3), \quad
\lambda_1 \ge \lambda_2 \ge \lambda_3 \ge 0
$$

**Mean Diffusivity (MD):**
$$
\mathrm{MD} = \frac{\lambda_1 + \lambda_2 + \lambda_3}{3} = \frac{\mathrm{tr}(\mathbf{D})}{3}
$$

**Fractional Anisotropy (FA)** — normalized variance of the eigenvalues:
$$
\mathrm{FA} = \sqrt{\frac{3}{2}}\,
\frac{\sqrt{(\lambda_1 - \mathrm{MD})^2 + (\lambda_2 - \mathrm{MD})^2 + (\lambda_3 - \mathrm{MD})^2}}
{\sqrt{\lambda_1^2 + \lambda_2^2 + \lambda_3^2}}
\in [0, 1]
$$

**Axial / Radial Diffusivity:**
$$
\mathrm{AD} = \lambda_1, \qquad
\mathrm{RD} = \frac{\lambda_2 + \lambda_3}{2}
$$

### 3. RK4 Streamline Integration (FACT / Euler fallback)

Fiber pathways are integral curves of the principal eigenvector field $\mathbf{v}_1(\mathbf{x})$:

$$
\frac{d\mathbf{r}}{ds} = \mathbf{v}_1\bigl(\mathbf{r}(s)\bigr)
$$

**4th-order Runge–Kutta** with trilinear vector-field interpolation and antipodal sign disambiguation:

$$
\begin{aligned}
\mathbf{k}_1 &= \mathbf{v}_1(\mathbf{r}_n) \\
\mathbf{k}_2 &= \mathbf{v}_1\!\bigl(\mathbf{r}_n + \tfrac{h}{2}\mathbf{k}_1\bigr) \\
\mathbf{k}_3 &= \mathbf{v}_1\!\bigl(\mathbf{r}_n + \tfrac{h}{2}\mathbf{k}_2\bigr) \\
\mathbf{k}_4 &= \mathbf{v}_1\!\bigl(\mathbf{r}_n + h\,\mathbf{k}_3\bigr) \\
\mathbf{r}_{n+1} &= \mathbf{r}_n + \tfrac{h}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)
\end{aligned}
$$

**Stopping criteria:**
- $\mathrm{FA}(\mathbf{r}) < \tau_{\mathrm{FA}}$ (exit white matter)
- $\angle(\mathbf{v}_n, \mathbf{v}_{n+1}) > \theta_{\max}$ (curvature / crossing discontinuity)
- Exit volume bounds or exceed $N_{\max}$ steps

Bidirectional tracking from each seed: propagate $+\mathbf{v}_1$ and $-\mathbf{v}_1$, then concatenate.

### 4. Structural Connectome Graph Metrics

Streamline endpoints are assigned to parcellation ROIs. Edge weight $W_{ij}$ = fiber count between regions $i$ and $j$.

**Density:**
$$
\delta = \frac{2\,|E|}{N(N-1)}
$$

**Clustering coefficient** (node $i$ with degree $k_i$ and $e_i$ edges among neighbors):
$$
C_i = \frac{2 e_i}{k_i(k_i - 1)}, \qquad C = \frac{1}{N}\sum_i C_i
$$

**Global efficiency** (via Floyd–Warshall shortest paths $d_{ij}$):
$$
E_{\mathrm{glob}} = \frac{1}{N(N-1)}\sum_{i \ne j}\frac{1}{d_{ij}}
$$

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRACTOFLOW PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│  phantom.py                                                     │
│  ┌──────────────────┐    Stejskal-Tanner + Rician noise         │
│  │ Synthetic DWI    │──────────────────────────────────────┐    │
│  │ Curved + Straight│   bvecs / bvals / mask / GT fibers   │    │
│  │ Bundle Phantom   │                                      │    │
│  └──────────────────┘                                      ▼    │
│  dti.py                                              ┌──────────┐│
│  ┌──────────────────┐   OLS design matrix B⁺         │ 4D DWI   ││
│  │ Tensor Fitting   │◄───────────────────────────────│ Volume   ││
│  │ FA / MD / AD / RD│   eigh → λ, V                  └──────────┘│
│  │ v₁ field         │──────────────────────────────────────┐    │
│  └──────────────────┘                                      │    │
│  tractography.py                                           ▼    │
│  ┌──────────────────┐   RK4 / Euler + trilinear interp  ┌──────┐│
│  │ Streamline       │◄──────────────────────────────────│ v₁,FA││
│  │ Tracker (bi-dir) │   FA thresh · max angle · bounds  └──────┘│
│  │ Tractogram       │──────────────────────────────────────┐    │
│  └──────────────────┘                                      │    │
│  connectome.py                                             ▼    │
│  ┌──────────────────┐   ROI endpoint mapping           ┌───────┐│
│  │ Connectome Graph │◄─────────────────────────────────│ fibers││
│  │ W, C, E_glob, δ  │   Floyd-Warshall · clustering    └───────┘│
│  └──────────────────┘                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Complexity Analysis

| Stage | Time | Space | Notes |
|-------|------|-------|-------|
| Design matrix construction | $O(K)$ | $O(K)$ | $K$ = # gradient directions |
| DTI OLS fit (masked) | $O(M \cdot K)$ | $O(M)$ | $M$ = # masked voxels; pinv precomputed $O(K^3)$ once |
| Eigendecomposition | $O(M)$ | $O(M)$ | batched `eigh` on $3\times3$ SPD tensors |
| Trilinear interpolation | $O(1)$ | $O(1)$ | 8-corner sample |
| RK4 single step | $O(1)$ | $O(1)$ | 4 field evaluations |
| Bidirectional tracking (1 seed) | $O(N_{\max})$ | $O(N_{\max})$ | typically $N_{\max} \le 500$ |
| Ensemble ($S$ seeds) | $O(S \cdot N_{\max})$ | $O(S \cdot \bar{L})$ | embarrassingly parallel |
| Connectome Floyd–Warshall | $O(N_r^3)$ | $O(N_r^2)$ | $N_r$ = # ROI nodes (typically 6–1000) |

---

## Quickstart

```bash
# Clone & install
git clone https://github.com/Sunzzx/tractoflow-diffusion-tensor-fiber-tracking-engine.git
cd tractoflow-diffusion-tensor-fiber-tracking-engine
pip install -e ".[dev]"

# Run the interactive terminal demo (ANSI FA map + streamline canvas)
python -m tractoflow.demo

# Run the full test suite
pytest -v
```

### Library Usage

```python
from tractoflow import (
    create_curved_bundle_phantom,
    fit_dti_volume,
    track_ensemble,
    build_parcellation_atlas,
    extract_connectome,
)

# 1. Synthetic multi-fiber phantom
phantom = create_curved_bundle_phantom(grid_size=(32, 32, 20), snr=40.0)

# 2. Fit DTI
dti = fit_dti_volume(phantom.dwi, phantom.bvecs, phantom.bvals, mask=phantom.mask)
print(f"Mean FA: {dti.fa[phantom.mask].mean():.3f}")

# 3. Track fibers from high-FA seeds
import numpy as np
seeds = [tuple(c.astype(float)) for c in np.argwhere(dti.fa > 0.3)[:100]]
v1 = dti.eigenvectors[..., :, 0]
tractogram = track_ensemble(v1, dti.fa, seeds, method="rk4", fa_threshold=0.15)
print(f"{tractogram.count} streamlines, mean length {tractogram.mean_length:.1f}")

# 4. Structural connectome
atlas = build_parcellation_atlas(phantom.dwi.shape[:3], n_regions=6)
graph = extract_connectome(tractogram, atlas)
print(f"Density={graph.density:.3f}  Clustering={graph.clustering_coefficient:.3f}")
print(f"Global efficiency={graph.global_efficiency:.4f}")
```

---

## Package Layout

```
tractoflow-diffusion-tensor-fiber-tracking-engine/
├── src/tractoflow/
│   ├── __init__.py          # Public API
│   ├── dti.py               # Stejskal-Tanner OLS tensor fitting + FA/MD/AD/RD
│   ├── phantom.py           # Synthetic curved + straight fiber DWI generator
│   ├── tractography.py      # RK4 / Euler bidirectional streamline tracking
│   ├── connectome.py        # Parcellation atlas + graph topology metrics
│   └── demo.py              # Interactive ANSI terminal visualization
├── tests/
│   └── test_tractoflow.py   # 25+ unit / integration / numerical tests
├── .github/workflows/ci.yml
├── pyproject.toml
├── LICENSE                  # MIT
└── README.md
```

---

## Benchmarks (Apple Silicon / CPython 3.12)

| Operation | Volume | Time |
|-----------|--------|------|
| Phantom generation (30 dir) | 28³×16 | ~1.2 s |
| DTI fit (masked OLS + eigh) | ~4k voxels | ~15 ms |
| RK4 ensemble (120 seeds) | — | ~80 ms |
| Connectome (6 ROIs) | — | <1 ms |
| Full demo pipeline | — | ~1.5 s |

---

## Commercial Licensing & Support

- **Open-source (MIT):** free for research, education, and commercial use
- **Enterprise support / custom clinical validation pipelines:** contact `ayushkandwal2007@gmail.com`
- **Sponsor development:** [GitHub Sponsors](https://github.com/sponsors/Sunzzx) · [Polar.sh](https://polar.sh/Sunzzx)
- **Stripe commercial license inquiries:** available on request for FDA/CE-adjacent deployments

---

## References

1. Basser, P. J., Mattiello, J., & LeBihan, D. (1994). MR diffusion tensor spectroscopy and imaging. *Biophysical Journal*, 66(1), 259–267.
2. Mori, S., et al. (1999). Three-dimensional tracking of axonal projections in the brain by magnetic resonance imaging. *Annals of Neurology*, 45(2), 265–269. (FACT)
3. Basser, P. J., et al. (2000). In vivo fiber tractography using DT-MRI data. *Magnetic Resonance in Medicine*, 44(4), 625–632.
4. Tournier, J. D., Calamante, F., & Connelly, A. (2012). MRtrix: Diffusion tractography in crossing fiber regions. *International Journal of Imaging Systems and Technology*.
5. Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity. *NeuroImage*, 52(3), 1059–1069.

---

## Author

**Ayush Kandwal** ([@Sunzzx](https://github.com/Sunzzx))  
Dual IIT Scholar · AI / Computational Neuroscience / Quantitative Research  
Built autonomously by the Hermes Frontier Research Engine.
