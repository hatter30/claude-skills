# CLAUDE.md — claude-skills Repository Guide

## Repository Overview

This repository hosts **Claude Skills** — modular, self-contained tools designed to extend Claude Code's capabilities via slash commands. Each skill lives in its own subdirectory and follows a consistent structure.

**Current skills:**
- `world-model-calculator/` — Video Diffusion model resource calculator (`/calc-video`)

---

## Repository Structure

```
claude-skills/
├── .claude/
│   └── settings.local.json        # Claude Code permission settings
├── world-model-calculator/        # Skill: video model resource calculator
│   ├── config.json                # Skill metadata, parameters, outputs schema
│   ├── SKILL.md                   # User-facing documentation
│   ├── CLAUDE.md                  # Technical notes for AI assistants
│   └── scripts/
│       └── calculator.py          # Python implementation (284 lines)
└── CLAUDE.md                      # This file
```

---

## Skill Structure Convention

Every skill directory must contain:

| File | Purpose |
|------|---------|
| `config.json` | Formal skill definition: name, trigger, version, parameters schema, outputs schema |
| `SKILL.md` | User-facing documentation with usage examples, parameter tables, sample output |
| `CLAUDE.md` | Technical notes: formulas, implementation decisions, AI assistant context |
| `scripts/` | Implementation code (language-agnostic) |

### `config.json` Schema

```json
{
  "name": "skill-name",
  "description": "One-line description",
  "version": "1.0.0",
  "trigger": "/command-name",
  "author": "claude-skills",
  "parameters": {
    "param_name": {
      "type": "string|integer|number|boolean",
      "required": true|false,
      "default": <value>,
      "description": "..."
    }
  },
  "outputs": {
    "output_name": "Description of what this output represents"
  }
}
```

---

## world-model-calculator Skill

### Purpose

Calculates computational resources for training and inference of Video Diffusion models (DiT/U-Net architectures) using Chinchilla-optimal compute allocation.

### Trigger

```
/calc-video
```

### CLI Usage

```bash
python world-model-calculator/scripts/calculator.py \
  -t <wall_time_hours> \
  -n <n_gpus> \
  -r <resolution> \
  -f <frames> \
  [--fps 24] \
  [-s 10] \
  [--gpu-type H100] \
  [--mfu 0.3] \
  [--gpu-price 2.0] \
  [--json]
```

**Required arguments:**
- `-t, --wall-time` — Training wall time in hours
- `-n, --n-gpu` — Number of GPUs
- `-r, --resolution` — Resolution: `480p`, `720p`, `1080p`, `2k`, `4k`, or `WxH`
- `-f, --frames` — Number of video frames

**Optional arguments:**
- `--fps` — Frames per second (default: 24)
- `-s, --steps` — Diffusion denoising steps (default: 10)
- `--gpu-type` — `A100` or `H100` (default: `H100`)
- `--mfu` — Model FLOPs utilization 0.0–1.0 (default: 0.3)
- `--gpu-price` — Cost in $/GPU-hour (default: 2.0)
- `--json` — Output results as JSON

### Core Formulas

```
gpu_hours  = wall_time × n_gpu
C_budget   = gpu_hours × GPU_TFLOPS × MFU × 3600
N_opt      = sqrt(C_budget / 120 / steps)      # Chinchilla-optimal model size
D_opt      = 20 × N_opt                         # Optimal training data tokens

# Latent dimensions (after VAE)
latent_h   = resolution_h / 8
latent_w   = resolution_w / 8
latent_t   = frames / 4
seq_len    = latent_h × latent_w × latent_t

# Memory (FSDP sharding)
M_model_per_gpu  = 16 × N / n_gpu
M_activation     = batch × seq_len × hidden_dim × num_layers × 2 bytes
hidden_dim       = sqrt(N / (12 × num_layers))

# Batch size constraints
batch = max multiple of 8 (Tensor Core) AND n_gpu (data parallel) that fits in VRAM
```

### GPU Specifications

| GPU  | BF16 TFLOPS | VRAM  |
|------|-------------|-------|
| A100 | 312         | 80 GB |
| H100 | 989         | 80 GB |

---

## Development Workflows

### Running a Calculation

```bash
# Basic example: 40h wall time, 8x H100, 720p, 50 frames
python world-model-calculator/scripts/calculator.py -t 40 -n 8 -r 720p -f 50

# JSON output for scripting
python world-model-calculator/scripts/calculator.py -t 40 -n 8 -r 1080p -f 120 --json
```

### Adding a New Skill

1. Create `<skill-name>/` directory
2. Add `config.json` following the schema above
3. Add `scripts/<implementation>` — pure implementations with no external deps preferred
4. Add `SKILL.md` with user-facing docs (examples, parameter tables)
5. Add `CLAUDE.md` with technical context (formulas, design decisions)

### Modifying `calculator.py`

Key sections:
- `GPU_SPECS` dict (line ~30): Add new GPU types here
- `VideoModelConfig` dataclass (line ~40): Add new config fields
- `calculate()` function: Core Chinchilla logic — handle numerical precision carefully
- `parse_resolution()`: Add new resolution presets here

---

## Code Conventions (Python)

- **Type hints** on all function signatures
- **Dataclasses** for configuration objects (immutable, validated)
- **UPPERCASE** constants: `GPU_SPECS`, `VAE_SPATIAL_DOWN`, `VAE_TEMPORAL_DOWN`
- **snake_case** for functions and variables
- **Docstrings** on module, classes, and public functions
- No external dependencies — stdlib only
- Memory values computed in bytes, displayed in GB (`/ 1024**3`)
- Batch size aligned to multiples of 8 (Tensor Core) and `n_gpu` (data parallel)
- Use `sys.exit(1)` for invalid inputs with a descriptive error message

---

## Permissions & Safety

`.claude/settings.local.json` restricts Claude Code to:
- `Bash(dir:*)` — directory listing
- `Bash(tree:*)` — tree view
- `Bash(python:*)` — running Python scripts

Do **not** expand these permissions unless a new skill explicitly requires it. Prefer skills that run with these minimal permissions.

---

## Git Workflow

- **Main branch:** `main`
- **Feature branches:** `claude/<description>-<id>` (created automatically by Claude Code)
- All development happens on feature branches; merge to `main` via PR
- Commit messages should be clear and describe *what changed and why*
- Remote: `https://github.com/hatter30/claude-skills`

---

## No CI/CD Currently

There are no automated tests or CI pipelines. When adding tests, place them in `<skill-name>/tests/` and prefer `pytest`. A `pytest.ini` or `pyproject.toml` at the repo root would be the appropriate place for test configuration.
