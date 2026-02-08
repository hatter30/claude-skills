---
name: world-model-calculator
description: Video Diffusion 모델(DiT/U-Net)의 학습/추론에 필요한 자원량(VRAM, FLOPs, GPU 시간)을 계산합니다.
---

# Video Model Resource Calculator

Video Diffusion 모델 학습을 위한 Chinchilla-optimal 설정 계산기

## 기능

주어진 학습 시간과 GPU로 최적 설정 계산:
- **N_opt**: Chinchilla-optimal 모델 크기
- **D_opt**: 최적 데이터량 (patches)
- **batch**: VRAM에 맞는 최적 배치 사이즈 (8 & n_gpu의 배수)
- **cost**: 예상 비용

## 사용법

```bash
# 기본 사용
python scripts/calculator.py -t 40 -n 8 -r 720p -f 50

# 상세 설정
python scripts/calculator.py -t 40 -n 8 -r 720p -f 50 --fps 10 --mfu 0.4 --gpu-type H100

# JSON 출력
python scripts/calculator.py -t 40 -n 8 -r 720p -f 50 --json
```

## 입력 파라미터

| 파라미터 | 설명 | 기본값 | 예시 |
|---------|------|-------|------|
| `-t, --wall-time` | 학습 시간 (hours) | (필수) | 40, 100, 240 |
| `-n, --n-gpu` | GPU 개수 | (필수) | 4, 8, 16 |
| `-r, --resolution` | 비디오 해상도 | (필수) | 720p, 1080p, 4k |
| `-f, --frames` | 프레임 수 | (필수) | 50, 120, 240 |
| `--fps` | 초당 프레임 | 24 | 10, 24, 30 |
| `-s, --steps` | 디퓨전 스텝 수 | 10 | 10, 20, 50 |
| `--gpu-type` | GPU 종류 | H100 | A100, H100 |
| `--mfu` | MFU (0.0-1.0) | 0.3 | 0.3, 0.4, 0.5 |
| `--gpu-price` | $/GPU-hour | 2.0 | 2.0, 3.0 |

## 해상도 프리셋

| 프리셋 | 해상도 |
|-------|--------|
| `480p` | 854x480 |
| `720p` | 1280x720 |
| `1080p` | 1920x1080 |
| `2k` | 2560x1440 |
| `4k` | 3840x2160 |

## 핵심 공식

### Chinchilla Optimal
```
gpu_hours = wall_time * n_gpu
C_budget = gpu_hours * FLOPS * MFU * 3600
N_opt = sqrt(C_budget / 120 / steps)
D_opt = 20 * N
```

### 메모리 계산
```
M_model_per_gpu = 16N / n_gpu  (FSDP)
M_activation = batch * seq_len * hidden_dim * num_layers * 2 bytes
```

### 배치 사이즈
```
batch = (max_batch // 8) * 8      # Tensor Core 최적화
batch = (batch // n_gpu) * n_gpu  # Data Parallel 균등 분배
```

### Video Diffusion 특화
```
seq_len = latent_h * latent_w * latent_frames
latent_h = height / 8 / patch_size
latent_w = width / 8 / patch_size
latent_frames = frames / 4
hidden_dim = sqrt(N / (12 * num_layers))
```

## GPU 스펙

| GPU | BF16 TFLOPS | VRAM |
|-----|-------------|------|
| A100 | 312 | 80 GB |
| H100 | 989 | 80 GB |

## 출력 예시

```
============================================================
Video Model Resource Calculator
============================================================

[Input]
  Wall time:           40 hours (1.7 days)
  n_gpu:               8
  Diffusion steps:     10

[Hardware]
  GPU:                 8x H100
  MFU:                 40%
  GPU-hours:           320

[Chinchilla Optimal]
  N_opt:               616.3M (hidden_dim=1,462)
  D_opt:               12.3B patches = 20 * N
  Training data:       396 hours of video

[Memory per GPU]
  Model+Optimizer:     1.1 GB (FSDP)
  Activations:         45.2 GB (batch=16)
  Total:               46.3 GB

[Results]
  Batch size:          16
  Cost:                $640
============================================================
```
