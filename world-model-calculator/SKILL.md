# Video Model Resource Calculator

Video Diffusion 모델(DiT, U-Net 기반)의 학습/추론에 필요한 자원량을 계산합니다.

## 기능

- **Latent Space 계산**: VAE 인코딩 후 잠재 공간 차원
- **Sequence Length**: DiT 어텐션을 위한 패치 수
- **Compute (FLOPs)**: 디퓨전 스텝당/비디오당 연산량
- **VRAM 계산**: 학습/추론 시 필요한 GPU 메모리
- **GPU 시간 추정**: A100 기준 추론 시간 및 학습 시간

## 사용법

```bash
# 기본 사용 (1080p, 5초 비디오)
python scripts/calculator.py --params 3B --resolution 1080p --frames 120

# 4K 고해상도
python scripts/calculator.py --params 8B --resolution 4k --frames 240 --steps 50

# 커스텀 해상도
python scripts/calculator.py --params 600M --resolution 1920x1080 --frames 60 --fps 30

# JSON 출력
python scripts/calculator.py --params 3B --resolution 1080p --frames 120 --json

# 도움말
python scripts/calculator.py --help
```

## 입력 파라미터

| 파라미터 | 설명 | 기본값 | 예시 |
|---------|------|-------|------|
| `--params`, `-p` | DiT/U-Net 파라미터 수 | (필수) | 600M, 3B, 8B |
| `--resolution`, `-r` | 비디오 해상도 | (필수) | 1080p, 4k, 1920x1080 |
| `--frames`, `-f` | 프레임 수 | (필수) | 60, 120, 240 |
| `--fps` | 초당 프레임 | 24 | 24, 30, 60 |
| `--steps`, `-s` | 디퓨전 스텝 수 | 30 | 20, 30, 50 |
| `--latent-channels` | VAE 잠재 채널 | 16 | 4, 16 |
| `--patch-size` | DiT 패치 크기 | 2 | 1, 2, 4 |
| `--precision` | 연산 정밀도 | bf16 | fp32, fp16, bf16 |
| `--batch-size`, `-b` | 배치 사이즈 | 1 | 1, 2, 4 |
| `--vae-params` | VAE 파라미터 수 | 83M | 83M, 200M |

## 해상도 프리셋

| 프리셋 | 해상도 |
|-------|--------|
| `480p` | 854x480 |
| `720p` | 1280x720 |
| `1080p` | 1920x1080 |
| `2k` | 2560x1440 |
| `4k` | 3840x2160 |

## 계산 공식

### Latent Space
```
latent_h = height / 8    (VAE spatial downsampling)
latent_w = width / 8
latent_frames = frames / 4    (temporal downsampling)
```

### Sequence Length (DiT)
```
seq_len = (latent_h / patch_size) × (latent_w / patch_size) × latent_frames
```

### FLOPs per Diffusion Step
```
FLOPs = 2 × params × seq_len + 4 × seq_len² × hidden_dim
        (transformer)          (attention)
```

### VRAM (Training)
```
VRAM = Model Weights + VAE + Optimizer States + Gradients + Activations
```

## 스케일링 특성

- **해상도**: Attention이 O(seq²)로 스케일 → 해상도 2배 = 메모리 4배
- **프레임 수**: 선형적으로 sequence length 증가
- **디퓨전 스텝**: 추론 시간에 선형 영향, VRAM에는 영향 없음
- **모델 크기**: VRAM과 compute 모두 선형 증가

## 참고 모델

| 모델 | 파라미터 | 해상도 | 최대 길이 |
|-----|---------|--------|----------|
| Sora (추정) | ~3B | 1080p | 60s |
| Runway Gen-3 | ~1-3B | 720p-1080p | 10s |
| Pika | ~1B | 1080p | 4s |
| Stable Video | 1.5B | 576x1024 | 4s |
