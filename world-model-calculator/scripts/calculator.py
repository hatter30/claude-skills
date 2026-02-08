#!/usr/bin/env python3
"""
Video Model Resource Calculator

wall_time * n_gpu로 Chinchilla-optimal 모델 크기 계산:
    gpu_hours = wall_time * n_gpu
    C_budget = gpu_hours * FLOPS * MFU * 3600
    N_opt = sqrt(C_budget / 120 / steps)
    batch = max multiple of 8 & n_gpu that fits in VRAM
"""

import argparse
import re
import sys
from dataclasses import dataclass


@dataclass
class VideoModelConfig:
    """Video model configuration."""

    wall_time_hours: int  # Training wall time in hours
    n_gpu: int  # Number of GPUs
    resolution: tuple[int, int]  # (width, height)
    frames: int  # Number of frames
    fps: int  # Frames per second
    diffusion_steps: int  # Denoising steps
    patch_size: int  # Patch size for DiT (default: 2)
    gpu_type: str  # GPU type (A100, H100)
    mfu: float  # Model FLOPS Utilization (0.0-1.0)
    gpu_price: float  # $/GPU-hour


# GPU specs
GPU_SPECS = {
    "A100": {"tflops_bf16": 312, "vram_gb": 80},
    "H100": {"tflops_bf16": 989, "vram_gb": 80},
}

# VAE downsampling factors
VAE_SPATIAL_DOWN = 8
VAE_TEMPORAL_DOWN = 4


def parse_resolution(value: str) -> tuple[int, int]:
    """Parse resolution like '1920x1080' or '1080p'."""
    presets = {
        "480p": (854, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "2k": (2560, 1440),
        "4k": (3840, 2160),
    }
    value = value.strip().lower()
    if value in presets:
        return presets[value]
    if match := re.match(r"^(\d+)x(\d+)$", value):
        return (int(match.group(1)), int(match.group(2)))
    raise ValueError(f"Cannot parse resolution: {value}")


def format_number(value: float) -> str:
    """Format with K/M/B/T suffixes."""
    for suffix, threshold in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if value >= threshold:
            return f"{value/threshold:.1f}{suffix}"
    return f"{value:.0f}"


def calculate_sequence_length(config: VideoModelConfig) -> int:
    """Calculate attention sequence length (Video 특화)."""
    w, h = config.resolution
    latent_h = h // VAE_SPATIAL_DOWN // config.patch_size
    latent_w = w // VAE_SPATIAL_DOWN // config.patch_size
    latent_frames = max(1, config.frames // VAE_TEMPORAL_DOWN)
    return latent_h * latent_w * latent_frames


def calculate(config: VideoModelConfig) -> dict:
    """
    wall_time * n_gpu → Optimal N, batch 계산

    gpu_hours = wall_time * n_gpu
    C_budget = gpu_hours * FLOPS_peak * MFU * 3600
    N_opt = sqrt(C_budget / (120 * steps))
    batch = max multiple of 8 & n_gpu that fits in VRAM
    """
    steps = config.diffusion_steps
    mfu = config.mfu
    n_gpu = config.n_gpu
    wall_time_hours = config.wall_time_hours
    gpu_spec = GPU_SPECS[config.gpu_type]
    flops_peak = gpu_spec["tflops_bf16"] * 1e12
    vram_gb = gpu_spec["vram_gb"]

    # GPU-hours = wall_time * n_gpu
    gpu_hours = wall_time_hours * n_gpu

    # Sequence length (Video 특화: 비디오당 패치 수)
    seq_len = calculate_sequence_length(config)

    # N_opt = sqrt(C_budget / (120 * steps))
    C_budget = gpu_hours * flops_peak * mfu * 3600
    N = int((C_budget / (120 * steps)) ** 0.5)

    # D_opt = 20 * N (Chinchilla optimal patches)
    D_opt = 20 * N

    # Training data hours (역산)
    num_videos = D_opt / seq_len
    video_duration = config.frames / config.fps
    total_video_hours = num_videos * video_duration / 3600

    C = 6 * N * D_opt * steps  # = 120 * N^2 * steps (since D_opt = 20N)

    # Memory calculation
    # Model + optimizer per GPU (FSDP): 16N / n_gpu
    M_model_per_gpu = 16 * N / n_gpu
    M_model_per_gpu_gb = M_model_per_gpu / (1024**3)

    # Activation per sample: seq_len * hidden_dim * num_layers * 2 bytes * factor
    # hidden_dim ≈ sqrt(N / (12 * num_layers)) for standard transformer
    num_layers = 24
    hidden_dim = int((N / (12 * num_layers)) ** 0.5)
    activation_per_sample = (
        seq_len * hidden_dim * num_layers * 2
    )  # bytes (with grad checkpointing)

    # Available VRAM for activations (use 80% of remaining)
    available_for_activation = (vram_gb * 1024**3 - M_model_per_gpu) * 0.8

    # Max batch size (8의 배수 & n_gpu의 배수)
    max_batch = int(available_for_activation / activation_per_sample)
    batch = (max_batch // 8) * 8  # 8의 배수 (Tensor Core)
    batch = (batch // n_gpu) * n_gpu  # n_gpu의 배수 (Data Parallel)
    batch = max(batch, n_gpu)  # 최소 n_gpu

    # Activation memory
    M_activation_gb = (batch * activation_per_sample) / (1024**3)
    M_total_per_gpu_gb = M_model_per_gpu_gb + M_activation_gb

    # Cost
    cost = gpu_hours * config.gpu_price

    return {
        "N": N,
        "hidden_dim": hidden_dim,
        "seq_len": seq_len,
        "D_opt": D_opt,
        "total_video_hours": total_video_hours,
        "C": C,
        "gpu_hours": gpu_hours,
        "batch": batch,
        "M_model_per_gpu_gb": M_model_per_gpu_gb,
        "M_activation_gb": M_activation_gb,
        "M_total_per_gpu_gb": M_total_per_gpu_gb,
        "cost": cost,
        "video_duration": video_duration,
    }


def print_results(config: VideoModelConfig, result: dict) -> None:
    """Print results."""
    w, h = config.resolution

    print("=" * 60)
    print("Video Model Resource Calculator")
    print("=" * 60)

    print(f"\n[Input]")
    print(
        f"  Wall time:           {config.wall_time_hours} hours ({config.wall_time_hours/24:.1f} days)"
    )
    print(f"  n_gpu:               {config.n_gpu}")
    print(f"  Diffusion steps:     {config.diffusion_steps}")

    print(f"\n[Sampling Target]")
    print(f"  Resolution:          {w}x{h}")
    print(
        f"  Duration:            {result['video_duration']:.1f}s ({config.frames} frames @ {config.fps}fps)"
    )
    print(f"  seq_len:             {result['seq_len']:,} patches")

    print(f"\n[Hardware]")
    print(f"  GPU:                 {config.n_gpu}x {config.gpu_type}")
    print(f"  MFU:                 {config.mfu:.0%}")
    print(f"  GPU-hours:           {result['gpu_hours']:,}")

    print(f"\n[Chinchilla Optimal]")
    print(
        f"  N_opt:               {format_number(result['N'])} (hidden_dim={result['hidden_dim']:,})"
    )
    print(f"  D_opt:               {format_number(result['D_opt'])} patches = 20 * N")
    print(f"  Training data:       {result['total_video_hours']:,.0f} hours of video")

    print(f"\n[Memory per GPU]")
    print(f"  Model+Optimizer:     {result['M_model_per_gpu_gb']:.1f} GB (FSDP)")
    print(
        f"  Activations:         {result['M_activation_gb']:.1f} GB (batch={result['batch']})"
    )
    print(f"  Total:               {result['M_total_per_gpu_gb']:.1f} GB")

    print(f"\n[Results]")
    print(f"  Batch size:          {result['batch']}")
    print(f"  Cost:                ${result['cost']:,.0f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Video Model Resource Calculator")
    parser.add_argument(
        "--wall-time", "-t", type=int, required=True, help="Training wall time in hours"
    )
    parser.add_argument("--n-gpu", "-n", type=int, required=True, help="Number of GPUs")
    parser.add_argument(
        "--resolution", "-r", required=True, help="Resolution (e.g., 1080p, 1920x1080)"
    )
    parser.add_argument(
        "--frames", "-f", type=int, required=True, help="Number of frames"
    )
    parser.add_argument("--fps", type=int, default=10, help="FPS (default: 10)")
    parser.add_argument(
        "--steps", "-s", type=int, default=10, help="Diffusion steps (default: 10)"
    )
    parser.add_argument(
        "--patch-size", type=int, default=2, help="Patch size (default: 2)"
    )
    parser.add_argument(
        "--gpu-type",
        choices=["A100", "H100"],
        default="H100",
        help="GPU type (default: H100)",
    )
    parser.add_argument(
        "--mfu", type=float, default=0.3, help="MFU 0.0-1.0 (default: 0.3)"
    )
    parser.add_argument(
        "--gpu-price", type=float, default=2.0, help="$/GPU-hour (default: 2.0)"
    )
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    try:
        resolution = parse_resolution(args.resolution)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    config = VideoModelConfig(
        wall_time_hours=args.wall_time,
        n_gpu=args.n_gpu,
        resolution=resolution,
        frames=args.frames,
        fps=args.fps,
        diffusion_steps=args.steps,
        patch_size=args.patch_size,
        gpu_type=args.gpu_type,
        mfu=args.mfu,
        gpu_price=args.gpu_price,
    )

    result = calculate(config)

    if args.json:
        import json

        input_data = {
            "wall_time_hours": config.wall_time_hours,
            "n_gpu": config.n_gpu,
            "resolution": f"{resolution[0]}x{resolution[1]}",
            "frames": config.frames,
            "steps": config.diffusion_steps,
            "gpu_type": config.gpu_type,
            "mfu": config.mfu,
        }
        print(json.dumps({"input": input_data, "result": result}, indent=2))
    else:
        print_results(config, result)


if __name__ == "__main__":
    main()
