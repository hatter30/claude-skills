#!/usr/bin/env python3
"""
Video Model Resource Calculator
Calculates required resources for Video Diffusion model training/inference.
Supports DiT (Diffusion Transformer) and U-Net based architectures.
"""

import argparse
import re
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class VideoModelConfig:
    """Video model configuration parameters."""
    params: int  # Model parameters (DiT/U-Net)
    resolution: tuple[int, int]  # (width, height)
    frames: int  # Number of frames
    fps: int  # Frames per second
    diffusion_steps: int  # Denoising steps
    latent_channels: int  # VAE latent channels (4 or 16)
    patch_size: int  # Patch size for DiT (8, 16, etc.)
    precision: str  # fp32, fp16, bf16
    batch_size: int
    vae_params: Optional[int] = None  # VAE parameters (optional)


@dataclass
class VideoResourceEstimate:
    """Estimated resource requirements."""
    # Latent info
    latent_size: tuple[int, int, int]  # (h, w, frames)
    sequence_length: int  # Total patches for attention
    video_duration: float  # seconds

    # Compute
    flops_per_step: float  # FLOPs per diffusion step
    flops_per_video: float  # Total FLOPs per video generation
    training_flops_per_video: float  # Training FLOPs (forward + backward)

    # VRAM
    training_vram_gb: float
    inference_vram_gb: float

    # Breakdown
    model_weights_gb: float
    vae_weights_gb: float
    latent_memory_gb: float
    attention_memory_gb: float

    # GPU time
    inference_time_a100: float  # seconds per video
    gpu_hours_1m_videos: float  # training 1M videos


# Bytes per parameter for each precision
BYTES_PER_PARAM = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
}

# A100 theoretical peak performance (TFLOPS)
A100_PEAK_TFLOPS = {
    "fp32": 19.5,
    "fp16": 312,
    "bf16": 312,
}

# Default VAE parameters (similar to SD VAE)
DEFAULT_VAE_PARAMS = 83_000_000  # ~83M for typical VAE
VAE_SPATIAL_DOWNSAMPLE = 8  # 8x spatial downsampling
VAE_TEMPORAL_DOWNSAMPLE = 4  # 4x temporal downsampling (for video VAE)


def parse_number(value: str) -> int:
    """Parse human-readable numbers like '7B', '600M'."""
    value = value.strip().upper()
    multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
    match = re.match(r'^([\d.]+)\s*([KMBT])?$', value)
    if not match:
        raise ValueError(f"Cannot parse number: {value}")
    num = float(match.group(1))
    suffix = match.group(2)
    if suffix:
        num *= multipliers[suffix]
    return int(num)


def parse_resolution(value: str) -> tuple[int, int]:
    """Parse resolution like '1920x1080' or '1080p'."""
    value = value.strip().lower()

    # Common presets
    presets = {
        "480p": (854, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "2k": (2560, 1440),
        "4k": (3840, 2160),
    }

    if value in presets:
        return presets[value]

    match = re.match(r'^(\d+)x(\d+)$', value)
    if match:
        return (int(match.group(1)), int(match.group(2)))

    raise ValueError(f"Cannot parse resolution: {value}. Use WxH or preset (480p, 720p, 1080p, 2k, 4k)")


def format_number(value: float, precision: int = 2) -> str:
    """Format large numbers with K/M/B/T suffixes."""
    if value >= 1e12:
        return f"{value/1e12:.{precision}f}T"
    elif value >= 1e9:
        return f"{value/1e9:.{precision}f}B"
    elif value >= 1e6:
        return f"{value/1e6:.{precision}f}M"
    elif value >= 1e3:
        return f"{value/1e3:.{precision}f}K"
    else:
        return f"{value:.{precision}f}"


def calculate_latent_dimensions(
    width: int,
    height: int,
    frames: int,
    spatial_downsample: int = VAE_SPATIAL_DOWNSAMPLE,
    temporal_downsample: int = VAE_TEMPORAL_DOWNSAMPLE,
) -> tuple[int, int, int]:
    """Calculate latent space dimensions after VAE encoding."""
    latent_h = height // spatial_downsample
    latent_w = width // spatial_downsample
    latent_frames = max(1, frames // temporal_downsample)
    return (latent_h, latent_w, latent_frames)


def calculate_sequence_length(
    latent_h: int,
    latent_w: int,
    latent_frames: int,
    patch_size: int,
) -> int:
    """Calculate total sequence length for DiT attention."""
    patches_h = latent_h // patch_size
    patches_w = latent_w // patch_size
    return patches_h * patches_w * latent_frames


def calculate_flops_per_step(
    params: int,
    sequence_length: int,
    hidden_dim: Optional[int] = None,
) -> float:
    """
    Calculate FLOPs per diffusion step.

    For transformer: ~2 * params * seq_len (matmuls)
    Plus attention: ~4 * seq_len^2 * hidden_dim
    """
    # Estimate hidden_dim from params if not provided
    # Rough estimate: params ≈ 12 * layers * hidden_dim^2
    if hidden_dim is None:
        # Assuming ~24 layers for typical DiT
        hidden_dim = int((params / (12 * 24)) ** 0.5)
        hidden_dim = max(hidden_dim, 1024)  # minimum 1024

    # Transformer forward pass
    transformer_flops = 2 * params * sequence_length

    # Self-attention (Q, K, V projections + attention + output)
    attention_flops = 4 * (sequence_length ** 2) * hidden_dim

    return transformer_flops + attention_flops


def calculate_vram_inference(
    params: int,
    vae_params: int,
    latent_h: int,
    latent_w: int,
    latent_frames: int,
    latent_channels: int,
    sequence_length: int,
    precision: str,
    batch_size: int,
) -> tuple[float, float, float, float, float]:
    """
    Calculate VRAM for inference.

    Returns: (total_gb, model_gb, vae_gb, latent_gb, attention_gb)
    """
    bytes_per_param = BYTES_PER_PARAM.get(precision, 2)

    # Model weights
    model_bytes = params * bytes_per_param
    model_gb = model_bytes / (1024**3)

    # VAE weights
    vae_bytes = vae_params * bytes_per_param
    vae_gb = vae_bytes / (1024**3)

    # Latent tensor: batch × channels × frames × h × w × bytes
    latent_elements = batch_size * latent_channels * latent_frames * latent_h * latent_w
    latent_bytes = latent_elements * bytes_per_param
    latent_gb = latent_bytes / (1024**3)

    # Attention memory (KV cache for all layers, estimate 24 layers)
    # Each layer: 2 * seq_len * hidden_dim * bytes (K and V)
    hidden_dim = int((params / (12 * 24)) ** 0.5)
    hidden_dim = max(hidden_dim, 1024)
    num_layers = 24
    attention_bytes = batch_size * num_layers * 2 * sequence_length * hidden_dim * bytes_per_param
    attention_gb = attention_bytes / (1024**3)

    total_gb = model_gb + vae_gb + latent_gb + attention_gb

    return total_gb, model_gb, vae_gb, latent_gb, attention_gb


def calculate_vram_training(
    params: int,
    vae_params: int,
    latent_h: int,
    latent_w: int,
    latent_frames: int,
    latent_channels: int,
    sequence_length: int,
    precision: str,
    batch_size: int,
) -> tuple[float, float, float, float, float]:
    """
    Calculate VRAM for training.

    Training requires: weights + optimizer states + gradients + activations
    """
    _, model_gb, vae_gb, latent_gb, attention_gb = calculate_vram_inference(
        params, vae_params, latent_h, latent_w, latent_frames,
        latent_channels, sequence_length, precision, batch_size
    )

    # Optimizer states (AdamW): 2 * params * 4 bytes (fp32 momentum + variance)
    optimizer_gb = (params * 4 * 2) / (1024**3)

    # Gradients: params * 4 bytes (fp32)
    gradients_gb = (params * 4) / (1024**3)

    # Activations: roughly 2-4x the attention memory for full backward pass
    activations_gb = attention_gb * 3

    total_gb = model_gb + vae_gb + optimizer_gb + gradients_gb + latent_gb + attention_gb + activations_gb

    return total_gb, model_gb, vae_gb, latent_gb, attention_gb


def estimate_inference_time(flops_per_video: float, precision: str) -> float:
    """Estimate inference time on A100 (seconds per video)."""
    peak_tflops = A100_PEAK_TFLOPS.get(precision, 312)
    # Assume 40% utilization for inference
    effective_tflops = peak_tflops * 0.40
    effective_flops_per_sec = effective_tflops * 1e12
    return flops_per_video / effective_flops_per_sec


def estimate_training_gpu_hours(training_flops: float, num_videos: int, precision: str) -> float:
    """Estimate GPU hours for training N videos."""
    peak_tflops = A100_PEAK_TFLOPS.get(precision, 312)
    # Assume 30% utilization for training
    effective_tflops = peak_tflops * 0.30
    effective_flops_per_sec = effective_tflops * 1e12
    total_flops = training_flops * num_videos
    seconds = total_flops / effective_flops_per_sec
    return seconds / 3600


def calculate_resources(config: VideoModelConfig) -> VideoResourceEstimate:
    """Calculate all resource requirements for a video model."""
    width, height = config.resolution
    vae_params = config.vae_params or DEFAULT_VAE_PARAMS

    # Latent dimensions
    latent_h, latent_w, latent_frames = calculate_latent_dimensions(
        width, height, config.frames
    )

    # Sequence length for attention
    seq_len = calculate_sequence_length(
        latent_h, latent_w, latent_frames, config.patch_size
    )

    # Video duration
    duration = config.frames / config.fps

    # FLOPs
    flops_per_step = calculate_flops_per_step(config.params, seq_len)
    flops_per_video = flops_per_step * config.diffusion_steps
    training_flops = flops_per_video * 3  # forward + backward ≈ 3x forward

    # VRAM
    training_vram, model_gb, vae_gb, latent_gb, attention_gb = calculate_vram_training(
        config.params, vae_params, latent_h, latent_w, latent_frames,
        config.latent_channels, seq_len, config.precision, config.batch_size
    )
    inference_vram, _, _, _, _ = calculate_vram_inference(
        config.params, vae_params, latent_h, latent_w, latent_frames,
        config.latent_channels, seq_len, config.precision, config.batch_size
    )

    # Time estimates
    inference_time = estimate_inference_time(flops_per_video, config.precision)
    gpu_hours_1m = estimate_training_gpu_hours(training_flops, 1_000_000, config.precision)

    return VideoResourceEstimate(
        latent_size=(latent_h, latent_w, latent_frames),
        sequence_length=seq_len,
        video_duration=duration,
        flops_per_step=flops_per_step,
        flops_per_video=flops_per_video,
        training_flops_per_video=training_flops,
        training_vram_gb=training_vram,
        inference_vram_gb=inference_vram,
        model_weights_gb=model_gb,
        vae_weights_gb=vae_gb,
        latent_memory_gb=latent_gb,
        attention_memory_gb=attention_gb,
        inference_time_a100=inference_time,
        gpu_hours_1m_videos=gpu_hours_1m,
    )


def print_results(config: VideoModelConfig, estimate: VideoResourceEstimate) -> None:
    """Print formatted results."""
    width = 80
    w, h = config.resolution

    print("=" * width)
    print("Video Model Resource Calculator".center(width))
    print("=" * width)

    print(f"Model Parameters:     {format_number(config.params)}")
    print(f"Resolution:           {w}x{h}")
    print(f"Frames:               {config.frames} ({estimate.video_duration:.1f}s @ {config.fps}fps)")
    print(f"Diffusion Steps:      {config.diffusion_steps}")
    print(f"Precision:            {config.precision}")
    print(f"Batch Size:           {config.batch_size}")
    print(f"Patch Size:           {config.patch_size}")
    print(f"Latent Channels:      {config.latent_channels}")

    print()
    print("-" * width)
    print("Latent Space".center(width))
    print("-" * width)

    lh, lw, lf = estimate.latent_size
    print(f"Latent Dimensions:    {lh} x {lw} x {lf} (H x W x Frames)")
    print(f"Sequence Length:      {estimate.sequence_length:,} patches")
    print(f"  (for attention computation)")

    print()
    print("-" * width)
    print("Compute Requirements".center(width))
    print("-" * width)

    print(f"FLOPs per Step:       {estimate.flops_per_step:.2e}")
    print(f"FLOPs per Video:      {estimate.flops_per_video:.2e} ({config.diffusion_steps} steps)")
    print(f"Training FLOPs:       {estimate.training_flops_per_video:.2e} (fwd + bwd)")

    print()
    print("-" * width)
    print("VRAM Requirements".center(width))
    print("-" * width)

    print(f"Training VRAM:        {estimate.training_vram_gb:.2f} GB")
    print(f"  - DiT Weights:      {estimate.model_weights_gb:.2f} GB")
    print(f"  - VAE Weights:      {estimate.vae_weights_gb:.2f} GB")
    print(f"  - Latent Tensors:   {estimate.latent_memory_gb:.2f} GB")
    print(f"  - Attention/KV:     {estimate.attention_memory_gb:.2f} GB")
    print(f"  - Optimizer+Grad:   ~{estimate.training_vram_gb - estimate.inference_vram_gb:.2f} GB")
    print()
    print(f"Inference VRAM:       {estimate.inference_vram_gb:.2f} GB")

    print()
    print("-" * width)
    print("GPU Time Estimate (A100 80GB)".center(width))
    print("-" * width)

    print(f"Inference per Video:  {estimate.inference_time_a100:.1f} seconds")
    print(f"Videos per Hour:      {3600 / estimate.inference_time_a100:.0f}")
    print()
    print(f"Training 1M Videos:   {estimate.gpu_hours_1m_videos:,.0f} GPU-hours")
    print(f"                      {estimate.gpu_hours_1m_videos/24:,.0f} GPU-days")

    if estimate.gpu_hours_1m_videos > 1000:
        for num_gpus in [8, 64, 256, 1024]:
            days = estimate.gpu_hours_1m_videos / 24 / num_gpus
            if days >= 0.1:
                print(f"  With {num_gpus} GPUs:      {days:,.1f} days")

    print()
    print("-" * width)
    print("Scaling Notes".center(width))
    print("-" * width)
    print("- Attention scales O(seq^2) - higher resolution increases cost quadratically")
    print("- More frames = longer sequences = higher memory & compute")
    print("- Fewer diffusion steps = faster but lower quality")
    print("=" * width)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate Video Diffusion Model resource requirements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --params 600M --resolution 1080p --frames 120
  %(prog)s --params 3B --resolution 4k --frames 300 --steps 50
  %(prog)s --params 8B --resolution 1920x1080 --frames 240 --fps 24
        """,
    )

    parser.add_argument(
        "--params", "-p",
        required=True,
        help="Model parameter count (e.g., 600M, 3B, 8B)",
    )

    parser.add_argument(
        "--resolution", "-r",
        required=True,
        help="Video resolution (e.g., 1920x1080, 1080p, 4k)",
    )

    parser.add_argument(
        "--frames", "-f",
        type=int,
        required=True,
        help="Number of frames to generate",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second (default: 24)",
    )

    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=30,
        help="Diffusion steps (default: 30)",
    )

    parser.add_argument(
        "--latent-channels",
        type=int,
        default=16,
        help="VAE latent channels (default: 16)",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=2,
        help="DiT patch size (default: 2)",
    )

    parser.add_argument(
        "--precision",
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Computation precision (default: bf16)",
    )

    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )

    parser.add_argument(
        "--vae-params",
        help="VAE parameter count (default: 83M)",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )

    args = parser.parse_args()

    try:
        params = parse_number(args.params)
        resolution = parse_resolution(args.resolution)
        vae_params = parse_number(args.vae_params) if args.vae_params else None
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    config = VideoModelConfig(
        params=params,
        resolution=resolution,
        frames=args.frames,
        fps=args.fps,
        diffusion_steps=args.steps,
        latent_channels=args.latent_channels,
        patch_size=args.patch_size,
        precision=args.precision,
        batch_size=args.batch_size,
        vae_params=vae_params,
    )

    estimate = calculate_resources(config)

    if args.json:
        import json
        result = {
            "config": {
                "params": config.params,
                "params_formatted": format_number(config.params),
                "resolution": f"{config.resolution[0]}x{config.resolution[1]}",
                "frames": config.frames,
                "fps": config.fps,
                "diffusion_steps": config.diffusion_steps,
                "precision": config.precision,
            },
            "estimate": {
                "latent_size": estimate.latent_size,
                "sequence_length": estimate.sequence_length,
                "video_duration_sec": estimate.video_duration,
                "flops_per_video": estimate.flops_per_video,
                "training_vram_gb": round(estimate.training_vram_gb, 2),
                "inference_vram_gb": round(estimate.inference_vram_gb, 2),
                "inference_time_sec": round(estimate.inference_time_a100, 2),
                "gpu_hours_1m_videos": round(estimate.gpu_hours_1m_videos, 0),
            },
        }
        print(json.dumps(result, indent=2))
    else:
        print_results(config, estimate)


if __name__ == "__main__":
    main()
