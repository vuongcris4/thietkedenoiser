# AGENTS.md — Guidelines for Agentic Coding in this Repository

## Project Overview
Pseudo-label Denoiser for Satellite Image Segmentation. Trains Denoising AutoEncoders (DAE) models to clean noisy pseudo-labels using OpenEarthMap dataset.

## Build / Setup Commands

```bash
# Create environment
python3 -m venv venv && source venv/bin/activate

# Install dependencies (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install segmentation-models-pytorch opencv-python matplotlib pyyaml tqdm

# Pull checkpoints (Git LFS)
git lfs pull
```

## Training Commands

```bash
cd src/

# Train DAE models (all use same script with different configs)
python train_dae.py --config ../configs/dae_lightweight.yaml   # Best: 97.78% mIoU
python train_dae.py --config ../configs/dae_resnet34.yaml
python train_dae.py --config ../configs/dae_effnet.yaml
python train_dae.py --config ../configs/dae_conditional.yaml

# Override config from CLI (dot notation)
python train_dae.py --config ../configs/dae_lightweight.yaml --override training.lr=0.0005

# Resume training
python train_dae.py --config ../configs/dae_lightweight.yaml --resume checkpoints/best.pth
```

## Evaluation Commands

```bash
# Evaluate single model
python evaluate_dae.py --checkpoint checkpoints/..._best.pth --model lightweight

# Full evaluation pipeline
python run_eval.py

# Demo inference (visualization)
python demo_inference.py --checkpoint checkpoints/..._best.pth
```

## Testing
No formal unit test framework exists. Validation is done via:
- Training script validates on held-out set each epoch
- Manual evaluation scripts (`evaluate_dae.py`, `run_eval.py`)
- Visual inspection via `demo_inference.py`

## Code Style Guidelines

### Imports
- Standard library imports first (`os`, `sys`, `json`, `time`)
- Third-party imports second (`numpy`, `torch`, `cv2`)
- Local imports last (relative to `src/`)
- Use `sys.path.insert(0, os.path.dirname(__file__))` for local imports in scripts
- Lazy imports for heavy dependencies (see `dae_model.py:_get_smp()`)

```python
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn

from config import load_config_from_args
from dae_model import build_model
```

### Formatting
- **Line length**: No strict limit, but ~100 chars preferred
- **Indentation**: 4 spaces (no tabs)
- **Blank lines**: 2 between top-level definitions, 1 between methods
- **Trailing whitespace**: Avoid

### Type Hints
- Use type hints for function signatures
- Common types: `int`, `float`, `str`, `bool`, `Optional[T]`, `Dict[K,V]`, `List[T]`, `Tuple[A,B]`
- Return types always specified

```python
def load_yaml(path: str) -> dict:
    ...

def _deep_merge(base: dict, override: dict) -> dict:
    ...
```

### Naming Conventions
- **Variables/functions**: `snake_case` (`noise_rate`, `load_config`)
- **Classes**: `PascalCase` (`NoiseGenerator`, `ConvBlock`, `DAEDataset`)
- **Constants**: `UPPER_CASE` (`NUM_CLASSES`, `CLASS_NAMES`)
- **Private helpers**: Prefix with underscore (`_get_region`, `_set_nested`, `_deep_merge`)

### Docstrings
- Module-level docstring at top explaining file purpose
- Function docstrings with Args/Returns for non-trivial functions
- Include usage examples for complex modules
- Vietnamese comments acceptable (mixed EN/VN style in codebase)

```python
def random_flip_noise(self, label: np.ndarray, noise_rate: float = 0.1) -> np.ndarray:
    """Random Flip: doi ngau nhien class cua pixels. Dom dom, khong cau truc."""
```

### Error Handling
- Use exceptions for exceptional cases
- Validate inputs at function entry for public APIs
- Graceful degradation for optional dependencies (e.g., wandb)

```python
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
```

### Class Structure
- Inherit from `nn.Module` for all model components
- Implement `__init__` and `forward` methods
- Use `nn.Sequential` for simple block compositions

```python
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        self.conv = nn.Sequential(...)
    
    def forward(self, x):
        return self.conv(x)
```

### Configuration Pattern
- Use YAML configs with inheritance (`_base_: default.yaml`)
- Support CLI overrides via dot notation (`--override training.lr=0.001`)
- Load configs via `config.py:load_config_from_args()`

### Logging & Debugging
- Print metrics every N batches during training (`if (batch_idx + 1) % 20 == 0`)
- Use `@torch.no_grad()` for validation/evaluation
- Log to Weights & Biases (W&B) when enabled

### File Organization
- All source code in `src/` directory
- Configs in `configs/` directory
- Checkpoints saved to `checkpoints/`
- Logs saved to `results/logs/`

## Architecture Notes

### DAE Models (All use dual-branch late fusion)
- Input: `rgb [B,3,H,W]` + `noisy_label [B,8,H,W]`
- Output: `logits [B,8,H,W]`
- Fusion at bottleneck with channel attention
- Skip connections from both branches at 4 scales

### Loss Function
```
Total Loss = CE×1.0 + Dice×1.0 + Boundary×0.5
```
Implemented in `dae_model.py:DAELoss`

## Common Pitfalls
1. **Pretrained weights**: Not helpful due to 11-channel input mismatch (ImageNet expects 3 channels)
2. **Model size**: Smaller models (12.82M) outperform larger ones (39M) in this domain
3. **Device**: Always use `device = 'cuda'` with `autocast()` for mixed precision

## Git Workflow
- Checkpoints tracked via Git LFS
- Data not tracked (downloaded separately)
- No formal branching strategy documented
