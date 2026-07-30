# TRELLIS.2 Seven-View vs Single-Image Demo on Google Colab (A100)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TylerOlszewski/TRELLIS.2/blob/main/notebooks/TRELLIS2_MultiImage_Colab_A100.ipynb)

Generate two **textured GLBs + turntable videos** of the same object for a direct comparison:

1. A seven-view run covering front-left, left, back-left, back, back-right, right, and
   front-right with this fork's
   [`run_multi_image()`](../trellis2/pipelines/trellis2_image_to_3d.py).
2. A single-image run using the held-out straight-on front image and the native `run()` method.

Neither path needs camera poses.

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
   The authentication cell prints the Hugging Face username and verifies file access to both
   gated models before loading the pipeline.
4. **Google Drive** with a few GB free — compiled CUDA wheels are cached there so you only pay
   the typical ~15–40 min build once. (Model weights ~15 GB are re-downloaded per session by default;
   flip `CACHE_MODELS_ON_DRIVE` in cell 2 if you'd rather keep them on Drive too.)

## Quick start

1. Click the **Open in Colab** badge above.
2. Select the A100 GPU **and 2025.10 runtime**, then run the cells top to bottom.
3. **First session only:** cell 6 downloads the matching official FlashAttention wheel, installs
   Eigen headers, then compiles nvdiffrast, nvdiffrec, CuMesh, FlexGEMM, and o-voxel at pinned revisions. Each wheel
   is cached immediately to `Drive/TRELLIS2_cache/wheels/`; rerunning after a disconnect resumes
   from the remaining package. Later sessions install successful builds from the cache in a few
   minutes.
4. In cell 8, upload exactly **seven non-front views** (or point
   `MULTIVIEW_DRIVE_FOLDER` at a folder containing them).
5. Run cells 9–12 to create `trellis2_7view.glb` and its turntable.
6. In cell 13, upload the held-out straight-on front image. Run cells 14–16 to create
   `trellis2_single_front.glb` and its turntable.
7. Both pairs of results are copied to `Drive/TRELLIS2_outputs/`.

To rerun a demo, rerun cells 8–12 (seven-view) or 13–16 (single image). The pipeline stays
loaded. Cells 12 and 16 park their finished mesh on CPU so the other demo gets a clean GPU;
rerunning the matching render or export cell moves it back automatically, so regeneration is
only needed when an input or generation parameter changes.

Cells 12 and 16 default to remeshing off. That selects O-Voxel's standard export path,
which removes duplicate faces, repairs non-manifold edges, drops small connected components,
fills small holes, and unifies face orientation. Set it to `True` only to test O-Voxel's
narrow-band dual-contouring remesher; this changes topology and can soften sharp details.

## Generation options

| Parameter | Cell | Values | Notes |
|---|---:|---|---|
| `MODE` | 10 | `stochastic` (default), `multidiffusion` | `stochastic` conditions each denoising step on a different view. `multidiffusion` averages all seven views at every step and is slower. |
| `RESOLUTION` / `SINGLE_RESOLUTION` | 10 / 14 | `default`, `512`, `1024`, `1024_cascade`, `1536_cascade` | `default` = the model config (`1024_cascade`). Drop to `512` on OOM. |
| `SEED` / `SINGLE_SEED` | 10 / 14 | any int | Defaults match at `42` for a controlled comparison. |
| `PREPROCESS` / `SINGLE_PREPROCESS` | 10 / 14 | on/off | Automatic background removal + recentering. Turn off only for clean-alpha inputs. |

### Choosing good views

- Use these seven views in cell 8: front-left, left, back-left, back, back-right, right,
  and front-right. Reserve straight-on front for cell 13.
- Prefix the seven filenames `01_` through `07_` in that order so the preview is easy to audit.
- Same object state, similar lighting in each shot — the views are aggregated without poses,
  so contradictory views average into blurry geometry.
- PNG, JPG/JPEG, WebP, HEIC, and HEIF are accepted directly. Images are resized internally
  (≤1024 px), so iPhone HEIC originals do not need to be converted first.

## What gets stored where

| Location | Contents | Safe to delete? |
|---|---|---|
| `Drive/TRELLIS2_cache/wheels/` | CUDA extension wheels + `env_tag.txt` | Yes — next run rebuilds them. Auto-cleared when the torch/CUDA/Python ABI changes. |
| `Drive/TRELLIS2_cache/hf_home/` | model weights (only if `CACHE_MODELS_ON_DRIVE`) | Yes — re-downloaded on demand. |
| `Drive/TRELLIS2_outputs/` | seven-view and single-image GLB/MP4 pairs | Your call. |

## Troubleshooting

See the table in the notebook's final cell. The two big ones:

- **Runtime verification fails**: select A100 and runtime version **2025.10**, then reconnect.
  The notebook deliberately stops instead of silently replacing PyTorch with an untested build.
- **CUDA 12.5 vs 12.6 warning**: expected on the pinned Colab image. Both share CUDA major
  version 12; cell 3 rejects unsafe major-version mismatches and includes both versions in the
  wheel-cache tag.
- **401/403 downloading models**: you haven't accepted the gated-model licenses, or the
  `HF_TOKEN` secret is missing/not shared with the notebook.
- **Seven-view upload is rejected**: cell 8 deliberately requires exactly seven supported
  image files. Keep the front image out of that folder/upload and use it in cell 13.
- **HEIC image is reported as unreadable**: rerun cell 4 to install `pillow-heif`, then rerun
  cell 7 to register the HEIF decoder before uploading images again.

## Running outside Colab

On any CUDA machine (e.g. a homelab server with a suitable GPU), skip the notebook and use the
repo directly — see [`setup.sh`](../setup.sh) for the environment and
[`example_multi_image.py`](../example_multi_image.py) for the same workflow as a CLI:

```bash
python example_multi_image.py \
  01_front_left.png 02_left.png 03_back_left.png 04_back.png \
  05_back_right.png 06_right.png 07_front_right.png \
  -o outputs --name trellis2_7view --mode stochastic
```

## Reproducibility notes

- The notebook pins the Colab runtime contract, FlashAttention 2.8.3 wheel, utils3d commit,
  and source revisions for every compiled third-party extension.
- Cached wheels are accepted only when PyTorch, its CUDA runtime, nvcc, Python, C++ ABI, and GPU
  architecture match the cache tag.
- Colab keeps past runtime versions available for a limited period. If 2025.10 is removed from
  the runtime selector, this notebook will need a deliberate dependency refresh rather than an
  automatic PyTorch replacement.
