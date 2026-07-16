"""
End-to-end multi-image 3D asset generation with TRELLIS.2.

Generates a single textured 3D asset from multiple images showing the SAME
object from different viewpoints. The views do not need camera poses; they are
aggregated at sampling time, either by cycling through them across denoising
steps ('stochastic') or by averaging their flow predictions at every step
('multidiffusion').

Example:
    python example_multi_image.py front.png side.png back.png -o outputs
    python example_multi_image.py views/*.png --mode multidiffusion --resolution 1536_cascade
"""
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a 3D asset (GLB + turntable video) from multiple views of the same object.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('images', nargs='+',
                        help="Paths to images of the same object from different viewpoints.")
    parser.add_argument('-o', '--output-dir', default='outputs_multi_image',
                        help="Directory to write the generated assets to.")
    parser.add_argument('--name', default='sample_multi',
                        help="Base name for the output files (<name>.glb, <name>.mp4).")
    parser.add_argument('--model', default='microsoft/TRELLIS.2-4B',
                        help="Pretrained pipeline, as a HuggingFace repo or local path.")
    parser.add_argument('--mode', choices=['stochastic', 'multidiffusion'], default='stochastic',
                        help="Multi-image aggregation mode. 'stochastic' cycles through the views across "
                             "denoising steps (fast); 'multidiffusion' averages all views at every step "
                             "(slower, more stable).")
    parser.add_argument('--resolution', choices=['512', '1024', '1024_cascade', '1536_cascade'], default=None,
                        help="Pipeline type. Defaults to the model config (1024_cascade for TRELLIS.2-4B).")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--no-preprocess', action='store_true',
                        help="Skip background removal / recentering (use if images already have clean alpha).")
    parser.add_argument('--no-video', action='store_true', help="Skip rendering the turntable video.")
    parser.add_argument('--no-glb', action='store_true', help="Skip exporting the GLB.")
    parser.add_argument('--envmap', default='assets/hdri/forest.exr',
                        help="HDRI environment map (EXR) used for the video render.")
    parser.add_argument('--texture-size', type=int, default=4096, help="GLB texture resolution.")
    parser.add_argument('--decimation-target', type=int, default=1000000,
                        help="Target face count for GLB decimation.")
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Can save GPU memory

    # Heavy imports after argparse so `--help` works everywhere.
    import cv2
    import imageio
    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.utils import render_utils
    from trellis2.renderers import EnvMap
    import o_voxel

    for path in args.images:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Input image not found: {path}")
    images = [Image.open(path) for path in args.images]
    print(f"Loaded {len(images)} view(s): {', '.join(args.images)}")
    if len(images) == 1:
        print("\033[93mWarning: only one image given; this is equivalent to single-image generation.\033[0m")

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    mesh = pipeline.run_multi_image(
        images,
        seed=args.seed,
        mode=args.mode,
        pipeline_type=args.resolution,
        preprocess_image=not args.no_preprocess,
    )[0]
    mesh.simplify(16777216)  # nvdiffrast limit

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.no_video:
        envmap = EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread(args.envmap, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        ))
        video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
        video_path = os.path.join(args.output_dir, f"{args.name}.mp4")
        imageio.mimsave(video_path, video, fps=15)
        print(f"Saved turntable video to {video_path}")

    if not args.no_glb:
        glb = o_voxel.postprocess.to_glb(
            vertices            =   mesh.vertices,
            faces               =   mesh.faces,
            attr_volume         =   mesh.attrs,
            coords              =   mesh.coords,
            attr_layout         =   mesh.layout,
            voxel_size          =   mesh.voxel_size,
            aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target   =   args.decimation_target,
            texture_size        =   args.texture_size,
            remesh              =   True,
            remesh_band         =   1,
            remesh_project      =   0,
            verbose             =   True
        )
        glb_path = os.path.join(args.output_dir, f"{args.name}.glb")
        glb.export(glb_path, extension_webp=True)
        print(f"Saved GLB to {glb_path}")


if __name__ == '__main__':
    main()
