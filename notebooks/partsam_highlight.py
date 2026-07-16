"""Run PartSAM on one TRELLIS GLB and export a color-highlighted part mesh.

PartSAM discovers part instances; it does not assign semantic names such as
"door" or "headlight". This wrapper keeps the original TRELLIS asset intact,
colors each discovered part on a decimated visualization copy, and writes a
JSON legend containing stable part IDs, colors, and face counts.

The inference flow follows PartSAM's MIT-licensed ``eval_everypart.py`` while
avoiding its optional Open3D and Apex dependencies and its hard-coded folders.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


def _farthest_point_sampling_fallback(points, num_samples: int, transpose: bool = False):
    """Pure-PyTorch fallback matching torkit3d's deterministic FPS interface."""
    import torch

    if transpose:
        points = points.transpose(1, 2)
    points = points.contiguous()
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected [B, N, 3] points, got {tuple(points.shape)}")
    if num_samples < 1 or num_samples > points.shape[1]:
        raise ValueError("FPS sample count must be between 1 and the number of points")

    batch_size, num_points, _ = points.shape
    indices = torch.empty((batch_size, num_samples), dtype=torch.long, device=points.device)
    min_distance = torch.full(
        (batch_size, num_points), float("inf"), dtype=points.dtype, device=points.device
    )
    farthest = torch.zeros(batch_size, dtype=torch.long, device=points.device)
    batch = torch.arange(batch_size, device=points.device)
    for index in range(num_samples):
        indices[:, index] = farthest
        centroid = points[batch, farthest].unsqueeze(1)
        distance = ((points - centroid) ** 2).sum(dim=-1)
        min_distance = torch.minimum(min_distance, distance)
        farthest = min_distance.max(dim=1).indices
    return indices


def _batch_index_select_fallback(input_tensor, index, dim: int):
    import torch

    if index.ndim == 1:
        index = index.unsqueeze(1)
        squeeze = True
    elif index.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"Expected a 1D or 2D batched index, got {index.ndim}D")
    if input_tensor.shape[0] != index.shape[0]:
        raise ValueError("Input and index batch sizes do not match")
    views = [1] * input_tensor.ndim
    views[0], views[dim] = index.shape[0], index.shape[1]
    expanded_shape = list(input_tensor.shape)
    expanded_shape[dim] = -1
    output = torch.gather(input_tensor, dim, index.view(views).expand(expanded_shape))
    return output.squeeze(dim) if squeeze else output


def _chamfer_distance_fallback(xyz1, xyz2, transpose=False, sqrt=False, eps=1e-12):
    import torch

    if xyz1.ndim == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.ndim == 2:
        xyz2 = xyz2.unsqueeze(0)
    if transpose:
        xyz1, xyz2 = xyz1.transpose(1, 2), xyz2.transpose(1, 2)
    distances = torch.cdist(xyz1, xyz2).square()
    first, second = distances.min(dim=2).values, distances.min(dim=1).values
    if sqrt:
        first = torch.sqrt(torch.clamp(first, eps))
        second = torch.sqrt(torch.clamp(second, eps))
    return first, second


def _partsam_fps():
    """Return torkit3d FPS, installing a compatible fallback when its build failed."""
    try:
        from torkit3d.ops.sample_farthest_points import sample_farthest_points

        return sample_farthest_points
    except (ImportError, OSError, RuntimeError):
        for name in list(sys.modules):
            if name == "torkit3d" or name.startswith("torkit3d."):
                del sys.modules[name]
        package_names = (
            "torkit3d",
            "torkit3d.nn",
            "torkit3d.ops",
        )
        for name in package_names:
            module = types.ModuleType(name)
            module.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = module

        functional = types.ModuleType("torkit3d.nn.functional")
        functional.batch_index_select = _batch_index_select_fallback
        fps_module = types.ModuleType("torkit3d.ops.sample_farthest_points")
        fps_module.sample_farthest_points = _farthest_point_sampling_fallback
        chamfer_module = types.ModuleType("torkit3d.ops.chamfer_distance")
        chamfer_module.chamfer_distance = _chamfer_distance_fallback
        sys.modules[functional.__name__] = functional
        sys.modules[fps_module.__name__] = fps_module
        sys.modules[chamfer_module.__name__] = chamfer_module
        print("torkit3d is unavailable; using the slower pure-PyTorch FPS fallback")
        return _farthest_point_sampling_fallback


def _load_mesh(path: Path):
    import trimesh

    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        try:
            loaded = loaded.to_geometry()
        except AttributeError:
            loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh in {path}, got {type(loaded).__name__}")
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError(f"Mesh contains no triangles: {path}")
    return loaded


def _texture_image(mesh) -> Any | None:
    material = getattr(getattr(mesh, "visual", None), "material", None)
    if material is None:
        return None
    image = getattr(material, "baseColorTexture", None)
    if image is None:
        image = getattr(material, "image", None)
    return image


def _sample_surface_with_color(mesh, count: int, seed: int):
    """Area-sample a mesh and interpolate its texture/vertex colors."""
    import trimesh

    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    total_area = float(areas.sum())
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("Cannot sample a mesh with zero surface area")

    rng = np.random.default_rng(seed)
    face_index = np.searchsorted(np.cumsum(areas), rng.random(count) * total_area)
    face_index = np.minimum(face_index, len(mesh.faces) - 1)
    triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)[face_index]]

    uvw = rng.random((count, 2))
    reflected = uvw.sum(axis=1) > 1.0
    uvw[reflected] = 1.0 - uvw[reflected]
    barycentric = np.column_stack((1.0 - uvw.sum(axis=1), uvw))
    points = np.einsum("ni,nij->nj", barycentric, triangles)

    colors = None
    visual = getattr(mesh, "visual", None)
    mesh_uv = getattr(visual, "uv", None)
    texture = _texture_image(mesh)
    if mesh_uv is not None and texture is not None:
        triangle_uv = np.asarray(mesh_uv)[np.asarray(mesh.faces)[face_index]]
        sample_uv = np.einsum("ni,nij->nj", barycentric, triangle_uv)
        colors = trimesh.visual.color.uv_to_interpolated_color(sample_uv, texture)

    if colors is None:
        vertex_colors = getattr(visual, "vertex_colors", None)
        if vertex_colors is not None and len(vertex_colors) == len(mesh.vertices):
            triangle_colors = np.asarray(vertex_colors)[np.asarray(mesh.faces)[face_index], :3]
            colors = np.einsum("ni,nij->nj", barycentric, triangle_colors)

    if colors is None:
        face_colors = getattr(visual, "face_colors", None)
        if face_colors is not None and len(face_colors) == len(mesh.faces):
            colors = np.asarray(face_colors)[face_index, :3]

    if colors is None:
        colors = np.full((count, 3), 192, dtype=np.uint8)
    colors = np.clip(np.asarray(colors)[:, :3], 0, 255).astype(np.uint8)
    return points.astype(np.float32), face_index.astype(np.int64), colors


def _normalize_for_partsam(mesh, count: int, seed: int, face_target: int):
    """Match PartSAM's inference normalization and retain its inverse."""
    import trimesh

    original_vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    bbmin, bbmax = original_vertices.min(axis=0), original_vertices.max(axis=0)
    original_extent = float((bbmax - bbmin).max())
    if original_extent <= 0:
        raise ValueError("Input mesh has a degenerate bounding box")
    original_center = (bbmin + bbmax) * 0.5
    original_scale = 1.8 / original_extent

    sampled_mesh = mesh.copy()
    sampled_mesh.vertices = (original_vertices - original_center) * original_scale
    points, sampled_faces, colors = _sample_surface_with_color(sampled_mesh, count, seed)
    normals = np.asarray(sampled_mesh.face_normals)[sampled_faces].astype(np.float32)

    geometry = trimesh.Trimesh(
        vertices=np.asarray(sampled_mesh.vertices).copy(),
        faces=np.asarray(sampled_mesh.faces).copy(),
        process=False,
    )
    if len(geometry.faces) > face_target:
        print(f"Decimating highlighted copy: {len(geometry.faces):,} -> {face_target:,} faces")
        geometry = geometry.simplify_quadric_decimation(face_count=face_target)

    sample_shift = (points.min(axis=0) + points.max(axis=0)) * 0.5
    points = points - sample_shift
    geometry.vertices = np.asarray(geometry.vertices) - sample_shift

    coord_min, coord_max = points.min(axis=0), points.max(axis=0)
    second_center = (coord_min + coord_max) * 0.5
    second_extent = float((coord_max - coord_min).max())
    if second_extent <= 0:
        raise ValueError("Sampled point cloud has a degenerate bounding box")
    second_scale = 1.8 / second_extent
    points = (points - second_center) * second_scale
    geometry.vertices = (np.asarray(geometry.vertices) - second_center) * second_scale

    inverse = {
        "original_center": original_center,
        "original_scale": original_scale,
        "sample_shift": sample_shift,
        "second_center": second_center,
        "second_scale": second_scale,
    }
    return geometry, points.astype(np.float32), colors, normals, inverse


def _restore_original_transform(mesh, inverse: dict[str, Any]) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices = vertices / inverse["second_scale"]
    vertices = vertices + inverse["second_center"] + inverse["sample_shift"]
    vertices = vertices / inverse["original_scale"] + inverse["original_center"]
    mesh.vertices = vertices


def _part_labels_from_masks(masks):
    import torch

    order = torch.argsort(masks.sum(dim=1), descending=True)
    masks = masks[order]
    labels = torch.full((masks.shape[1],), -1, dtype=torch.int64)
    for part_id, mask in enumerate(masks):
        labels[mask] = part_id
    return labels.numpy()


def _fill_uncovered_labels(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    valid = labels >= 0
    if not valid.any():
        raise RuntimeError("PartSAM accepted masks but none covered the sampled surface")
    if valid.all():
        return labels
    nearest = cKDTree(points[valid]).query(points[~valid], k=1)[1]
    labels = labels.copy()
    labels[~valid] = labels[valid][nearest]
    return labels


def _labels_for_faces(mesh, points: np.ndarray, point_labels: np.ndarray, neighbors: int):
    from scipy.spatial import cKDTree

    k = max(1, int(neighbors))
    nearest = cKDTree(points).query(np.asarray(mesh.triangles_center), k=k)[1]
    if k == 1:
        return point_labels[nearest]
    neighbor_labels = point_labels[np.asarray(nearest)]
    # K is intentionally small (3 by default); this avoids scipy.stats' version-sensitive API.
    output = np.empty(len(neighbor_labels), dtype=np.int64)
    for index, row in enumerate(neighbor_labels):
        values, counts = np.unique(row, return_counts=True)
        output[index] = values[np.argmax(counts)]
    return output


def _write_legend(mesh, path: Path, metadata: dict[str, Any]) -> int:
    face_colors = np.asarray(mesh.visual.face_colors, dtype=np.uint8)
    unique, counts = np.unique(face_colors, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    total = int(counts.sum())
    parts = []
    for part_id, color_index in enumerate(order):
        rgba = unique[color_index].tolist()
        face_count = int(counts[color_index])
        parts.append(
            {
                "part_id": part_id,
                "color_hex": "#" + "".join(f"{channel:02X}" for channel in rgba[:3]),
                "rgba": rgba,
                "face_count": face_count,
                "face_fraction": face_count / total,
                "semantic_name": None,
            }
        )
    payload = {
        "note": (
            "PartSAM discovers unlabeled part instances. Colors identify regions but do not "
            "automatically mean door, hood, window, headlight, or another semantic class."
        ),
        **metadata,
        "part_count": len(parts),
        "parts": parts,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return len(parts)


def run(args: argparse.Namespace) -> dict[str, str]:
    partsam_dir = args.partsam_dir.resolve()
    config_dir = partsam_dir / "configs"
    if not (partsam_dir / "PartSAM").is_dir() or not config_dir.is_dir():
        raise FileNotFoundError(f"PartSAM checkout is incomplete: {partsam_dir}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"PartSAM checkpoint not found: {args.checkpoint}")

    sys.path.insert(0, str(partsam_dir))

    import hydra
    import torch
    import trimesh
    from accelerate.utils import set_seed
    from omegaconf import OmegaConf
    from safetensors.torch import load_model

    sample_farthest_points = _partsam_fps()

    # PartSAM's utility module imports pointops globally even though the smoothing
    # functions used here do not call it. Avoid a second custom CUDA build.
    try:
        __import__("pointops")
    except ImportError:
        sys.modules["pointops"] = types.ModuleType("pointops")
    from utils.infer_utils import nms, post_processing  # type: ignore

    set_seed(args.seed)
    overrides = [
        f"eval_params.ckpt_path={args.checkpoint}",
        f"eval_params.iou_threshold={args.iou_threshold}",
        f"eval_params.nms_threshold={args.nms_threshold}",
        f"eval_params.threshold_percentage_size={args.min_part_fraction}",
        f"eval_params.threshold_percentage_area={args.min_part_fraction}",
        f"eval_params.use_graph_cut={str(args.graph_cut).lower()}",
    ]
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(config_name="partsam", overrides=overrides)
    OmegaConf.resolve(cfg)

    source_mesh = _load_mesh(args.input)
    display_mesh, points_np, colors_np, normals_np, inverse = _normalize_for_partsam(
        source_mesh,
        count=args.num_points,
        seed=args.seed,
        face_target=args.face_target,
    )

    print("Loading PartSAM checkpoint...")
    model = hydra.utils.instantiate(cfg.model)
    # Standard torch LayerNorm has checkpoint-compatible parameters; Apex's fused
    # replacement is an optional speed optimization and is not needed for inference.
    load_model(model, str(args.checkpoint))
    model.eval().cuda()

    coords = torch.from_numpy(points_np).cuda().contiguous()
    colors = torch.from_numpy(colors_np).float().cuda().div_(255.0).contiguous()
    normals = torch.from_numpy(normals_np).cuda().contiguous()
    vertices = torch.from_numpy(np.asarray(display_mesh.vertices)).float().cuda().contiguous()
    faces = torch.from_numpy(np.asarray(display_mesh.faces)).long().cuda().contiguous()

    prompt_indices = sample_farthest_points(coords.unsqueeze(0), args.fps_prompts)[0]
    effective_batch = min(args.batch_size, args.fps_prompts)
    all_masks, all_scores = [], []
    print(f"Running {args.fps_prompts} PartSAM prompts in batches of {effective_batch}...")
    with torch.no_grad():
        for start in range(0, args.fps_prompts, effective_batch):
            selected = prompt_indices[start : start + effective_batch]
            actual = len(selected)
            if actual < effective_batch:
                selected = torch.cat((selected, selected[:1].repeat(effective_batch - actual)))
            batch_input = {
                # The point-cloud embedding is shared by every prompt. Keeping the
                # source batch at one lets PartSAM repeat only the small prompt-side
                # tensors instead of encoding 100k points 32 times.
                "coords": coords.unsqueeze(0),
                "color": colors.unsqueeze(0),
                "normal": normals.unsqueeze(0),
                "point_to_face": torch.zeros((1, 1), dtype=torch.long, device="cuda"),
                "vertices": vertices.unsqueeze(0),
                "faces": faces.unsqueeze(0),
                "prompt_coords": coords[selected].unsqueeze(1),
                "selected_indices": selected,
                "prompt_labels": torch.ones(
                    (effective_batch, 1), dtype=torch.long, device="cuda"
                ),
            }
            batch_masks, batch_scores = model.predict_masks(**batch_input)
            all_masks.append(batch_masks[:actual].cpu())
            all_scores.append(batch_scores[:actual].cpu())
            del batch_input, batch_masks, batch_scores

    masks = torch.cat(all_masks, dim=0).reshape(-1, args.num_points) > 0
    scores = torch.cat(all_scores, dim=0).reshape(-1)
    accepted = scores > args.iou_threshold
    masks, scores = masks[accepted], scores[accepted]
    if len(masks) == 0:
        raise RuntimeError(
            "PartSAM found no masks above the IoU threshold; lower --iou-threshold "
            "(for example, from 0.65 to 0.55)."
        )
    print(f"Masks after score threshold: {len(masks)}")
    kept = nms(masks, scores, threshold=args.nms_threshold)
    masks = masks[kept]
    print(f"Masks after NMS: {len(masks)}")

    point_labels = _part_labels_from_masks(masks)
    point_labels = _fill_uncovered_labels(points_np, point_labels)
    face_labels = _labels_for_faces(display_mesh, points_np, point_labels, args.face_neighbors)

    display_mesh.visual = trimesh.visual.ColorVisuals(mesh=display_mesh)
    highlighted = post_processing(face_labels, display_mesh, cfg.eval_params)
    _restore_original_transform(highlighted, inverse)

    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    ply_path = output_prefix.with_suffix(".ply")
    glb_path = output_prefix.with_suffix(".glb")
    legend_path = output_prefix.with_name(output_prefix.name + "_legend").with_suffix(".json")
    highlighted.export(ply_path)
    highlighted.export(glb_path)
    part_count = _write_legend(
        highlighted,
        legend_path,
        {
            "source": str(args.input.resolve()),
            "highlighted_glb": str(glb_path),
            "highlighted_ply": str(ply_path),
            "partsam_revision": args.partsam_revision,
            "seed": args.seed,
            "num_points": args.num_points,
            "fps_prompts": args.fps_prompts,
            "iou_threshold": args.iou_threshold,
            "nms_threshold": args.nms_threshold,
            "face_target": args.face_target,
        },
    )

    del model, coords, colors, normals, vertices, faces
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Exported {part_count} highlighted regions to {glb_path}")
    return {"glb": str(glb_path), "ply": str(ply_path), "legend": str(legend_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Textured TRELLIS GLB")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--partsam-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partsam-revision", default="unknown")
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--num-points", type=int, default=100_000)
    parser.add_argument("--fps-prompts", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iou-threshold", type=float, default=0.65)
    parser.add_argument("--nms-threshold", type=float, default=0.30)
    parser.add_argument("--min-part-fraction", type=float, default=0.002)
    parser.add_argument("--face-target", type=int, default=100_000)
    parser.add_argument("--face-neighbors", type=int, default=3)
    parser.add_argument("--graph-cut", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.num_points < 1 or args.fps_prompts < 1 or args.batch_size < 1:
        raise ValueError("Point, prompt, and batch counts must all be positive")
    if args.fps_prompts > args.num_points:
        raise ValueError("--fps-prompts cannot exceed --num-points")
    if args.face_target < 1000:
        raise ValueError("--face-target must be at least 1000")
    run(args)


if __name__ == "__main__":
    main()
