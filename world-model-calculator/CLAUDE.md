# Video Model Resource Calculator

wall_time * n_gpu로 Chinchilla-optimal 모델 크기 및 배치 사이즈 계산

## 핵심 공식

```
gpu_hours = wall_time * n_gpu
C_budget = gpu_hours * FLOPS * MFU * 3600
N_opt = sqrt(C_budget / 120 / steps)
D_opt = 20 * N
batch = max multiple of 8 & n_gpu that fits in VRAM
```

## 사용법

```bash
python scripts/calculator.py -t 40 -n 8 -r 720p -f 50 --fps 10 --mfu 0.4
```

**필수 인자:**
- `-t, --wall-time`: 학습 시간 (hours)
- `-n, --n-gpu`: GPU 개수
- `-r, --resolution`: 해상도 (480p, 720p, 1080p, 2k, 4k)
- `-f, --frames`: 프레임 수

**선택 인자:**
- `--fps`: FPS (기본: 24)
- `-s, --steps`: Diffusion steps (기본: 10)
- `--gpu-type`: A100 또는 H100 (기본: H100)
- `--mfu`: MFU 0.0-1.0 (기본: 0.3)
- `--gpu-price`: $/GPU-hour (기본: 2.0)

## 메모리 계산

```
M_model_per_gpu = 16N / n_gpu  (FSDP)
M_activation = batch * seq_len * hidden_dim * num_layers * 2 bytes
batch = (max_batch // 8) * 8   # Tensor Core
batch = (batch // n_gpu) * n_gpu  # Data Parallel
```

## GPU 스펙

| GPU | BF16 TFLOPS | VRAM |
|-----|-------------|------|
| A100 | 312 | 80 GB |
| H100 | 989 | 80 GB |

## Video Diffusion 특화

- `seq_len = latent_h * latent_w * latent_frames`
- VAE downsampling: 8x spatial, 4x temporal
- `hidden_dim = sqrt(N / (12 * num_layers))`
