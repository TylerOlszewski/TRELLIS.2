# TRELLIS.2 Multi-Image → 3D on Google Colab (A100)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TylerOlszewski/TRELLIS.2/blob/main/notebooks/TRELLIS2_MultiImage_Colab_A100.ipynb)

Turn **multiple photos of the same object** (different angles, no camera poses needed) into a
single **textured GLB + turntable video**, using this fork's
[`run_multi_image()`](../trellis2/pipelines/trellis2_image_to_3d.py) pipeline on a Colab A100.

## Prerequisites

1. **Colab with an A100 and runtime version 2025.10** — Runtime → *Change runtime type* →
   GPU: **A100**, Runtime Version: **2025.10**. The pinned image provides Python 3.12 and
   PyTorch 2.8, matching the notebook's prebuilt FlashAttention wheel and compiled-extension
   cache. Do not use Colab's changing default runtime for this notebook.
2. **Hugging Face account** with access to two *gated* models (accept the license on each
   model page before starting Colab):
   - [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) — image encoder
   - [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) — background removal
3. **HF token in Colab Secrets** — create a *read* token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then in Colab open
   the **key icon** in the left sidebar and add it as `HF_TOKEN` (enable notebook access).
4. **Google Drive** with a few GB free — compiled CUDA wheels are cached there so you only pay
   the typical ~15–40 min build once. (Model weights ~15 GB are re-downloaded per session by default;
   flip `CACHE_MODELS_ON_DRIVE` in cell 2 if you'd rather keep them on Drive too.)

## Quick start

1. Click the **Open in Colab** badge above.
2. Select the A100 GPU **and 2025.10 runtime**, then run the cells top to bottom.
3. **First session only:** cell 6 downloads the matching official FlashAttention wheel, then
   compiles nvdiffrast, nvdiffrec, CuMesh, FlexGEMM, and o-voxel at pinned revisions. Each wheel
   is cached immediately to `Drive/TRELLIS2_cache/wheels/`; rerunning after a disconnect resumes
   from the remaining package. Later sessions install the cache in a few minutes.
4. In cell 8, upload **2–4 views of the same object** (or point `DRIVE_FOLDER` at a folder of
   images on your Drive).
5. Run cells 9–12: the pipeline loads once, generates the mesh, renders a turntable preview
   inline, and exports `trellis2_multiview.glb`. Results are also copied to
   `Drive/TRELLIS2_outputs/`.

## Generation options (cell 10)

| Parameter | Values | Notes |
|---|---|---|
| `MODE` | `stochastic` (default), `multidiffusion` | `stochastic` conditions each denoising step on a different view — fast, no extra compute. `multidiffusion` averages all views at every step — slower but more stable when views disagree. |
| `RESOLUTION` | `default`, `512`, `1024`, `1024_cascade`, `1536_cascade` | `default` = the model config (`1024_cascade`). Drop to `512` on OOM. |
| `SEED` | any int | Same seed + same inputs → same asset. |
| `PREPROCESS` | on/off | Automatic background removal + recentering. Turn off only if your images already have clean alpha. |

### Choosing good views

- 2–4 images covering different sides (front/side/back) works well; more views than
  denoising steps (12) get skipped in `stochastic` mode.
- Same object state, similar lighting in each shot — the views are aggregated without poses,
  so contradictory views average into blurry geometry.
- Any resolution; images are resized internally (≤1024 px).

## What gets stored where

| Location | Contents | Safe to delete? |
|---|---|---|
| `Drive/TRELLIS2_cache/wheels/` | CUDA extension wheels + `env_tag.txt` | Yes — next run rebuilds them. Auto-cleared when the torch/CUDA/Python ABI changes. |
| `Drive/TRELLIS2_cache/hf_home/` | model weights (only if `CACHE_MODELS_ON_DRIVE`) | Yes — re-downloaded on demand. |
| `Drive/TRELLIS2_outputs/` | your generated GLB/MP4 files | Your call. |

## Troubleshooting

See the table in the notebook's final cell. The two big ones:

- **Runtime verification fails**: select A100 and runtime version **2025.10**, then reconnect.
  The notebook deliberately stops instead of silently replacing PyTorch with an untested build.
- **CUDA 12.5 vs 12.6 warning**: expected on the pinned Colab image. Both share CUDA major
  version 12; cell 3 rejects unsafe major-version mismatches and includes both versions in the
  wheel-cache tag.
- **401/403 downloading models**: you haven't accepted the gated-model licenses, or the
  `HF_TOKEN` secret is missing/not shared with the notebook.

## Running outside Colab

On any CUDA machine (e.g. a homelab server with a suitable GPU), skip the notebook and use the
repo directly — see [`setup.sh`](../setup.sh) for the environment and
[`example_multi_image.py`](../example_multi_image.py) for the same workflow as a CLI:

```bash
python example_multi_image.py front.png side.png back.png -o outputs --mode stochastic
```

## Reproducibility notes

- The notebook pins the Colab runtime contract, FlashAttention 2.8.3 wheel, utils3d commit, and
  source revisions for every compiled third-party extension.
- Cached wheels are accepted only when PyTorch, its CUDA runtime, nvcc, Python, C++ ABI, and GPU
  architecture match the cache tag.
- Colab keeps past runtime versions available for a limited period. If 2025.10 is removed from
  the runtime selector, this notebook will need a deliberate dependency refresh rather than an
  automatic PyTorch replacement.
