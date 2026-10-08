# Environment Documentation

**Project:** Representation-Based Unsupervised Detection of Poisoning Attacks in Federated Learning  
**Date Recorded:** 2026-09-30  

> [!IMPORTANT]
> This file must be kept up-to-date. Every dependency change must be documented here with a rationale.

---

## 1. Hardware

| Property | Value |
|---|---|
| Operating System | macOS 26.6.2 (Sequoia) |
| Architecture | arm64 (Apple Silicon) |
| Chip | Apple M-series (confirmed via MPS availability) |
| CUDA GPU | Not available |
| MPS (Metal Performance Shaders) | **Available** — used as primary accelerator |

---

## 2. Software Versions

| Package | Version | Notes |
|---|---|---|
| Python | 3.14.6 | Homebrew, `/opt/homebrew/bin/python3` |
| pip | 26.2.1 | Inside venv |
| torch | **2.14.1** | CPU wheel from `download.pytorch.org/whl/cpu`; MPS enabled via macOS backend |
| torchvision | 0.29.1 | Paired with torch 2.14.1 |
| scikit-learn | 1.9.1 | K-Means, metrics, decomposition |
| numpy | 2.5.2 | Array operations, Dirichlet sampling |
| scipy | 1.18.1 | Statistical utilities, `median_abs_deviation` |
| pandas | 3.0.6 | Results storage, CSV logging |
| matplotlib | 3.11.2 | Plotting |
| seaborn | 0.13.2 | Statistical plots |
| pyyaml | 6.0.3 | Configuration files |
| tqdm | 4.70.1 | Progress bars |
| Pillow | 12.3.0 | Image loading (auto-installed with torchvision) |
| sympy | 1.14.0 | Auto-installed with torch |
| networkx | 3.6.1 | Auto-installed with torch |

---

## 3. Virtual Environment

```
Location: BTP-FL-Poisoning/.venv/
Created with: python3 -m venv .venv
Activate: source .venv/bin/activate
Python interpreter: .venv/bin/python
```

### Reproduce the Environment

```bash
# From BTP-FL-Poisoning/
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn scipy pandas matplotlib seaborn pyyaml tqdm
```

Or using the requirements file:
```bash
pip install -r requirements.txt
```

---

## 4. Device Strategy

This project runs on **Apple Silicon** with **MPS acceleration**.

### Device Selection Logic (Used in All Code)

```python
import torch

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
```

### Why MPS?

- Apple Silicon ARM64 provides hardware-accelerated GPU compute via Metal Performance Shaders
- MPS is significantly faster than CPU for tensor operations
- PyTorch 2.x has stable MPS support
- No CUDA GPU available on this machine

### MPS Limitations (Documented)

| Issue | Status | Workaround |
|---|---|---|
| Some operations fall back to CPU silently | Known | Acceptable for our use case |
| `torch.linalg.eig()` not supported on MPS | If encountered | Move tensor to CPU for that operation |
| MPS does not support `float64` fully | Known | Use `float32` by default |

---

## 5. Compatibility Issues with Reference Repositories

### 5.1 FedCVG

| Issue | Severity | Resolution |
|---|---|---|
| Comments in Chinese | Low | Documentation is ours; no code change needed |
| `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` | None | Already handles CPU correctly |
| `from sklearn.cluster import KMeans` — version compatibility | Low | scikit-learn 1.9.1 is compatible |
| Requires `tqdm`, `scikit-learn`, `scipy` | None | All installed |

**FedCVG Requirements (original):**
```
torch>=1.7.0        → We have 2.14.1 ✓
torchvision>=0.8.1  → We have 0.29.1 ✓
numpy>=1.19.2       → We have 2.5.2 ✓
Pillow>=8.0.0       → We have 12.3.0 ✓
tqdm>=4.50.2        → We have 4.70.1 ✓
scikit-learn>=0.24.0→ We have 1.9.1 ✓
```

**Status:** FedCVG is compatible with the current environment.

### 5.2 FedCC

| Issue | Severity | Resolution |
|---|---|---|
| **`to('cuda:0')` hardcoded** | **CRITICAL** | Do NOT modify FedCC source. Use device-agnostic reimplementation in `proposed/` |
| **No standalone `main.py`** | High | Wrap in our own script if needed for baseline comparison |
| PLR extracted from `keys[-4]` (fragile key indexing) | Medium | Our implementation uses explicit layer names |
| `from aggregate import multi_krum, krum` (relative import) | Low | Must run from `src/` directory |
| No requirements.txt | Medium | Manually inferred from imports |

**Inferred FedCC Requirements:**
```
torch       (any version supporting cuda) → We have 2.14.1
torchvision                               → We have 0.29.1
scikit-learn                              → We have 1.9.1
scipy                                     → We have 1.18.1 (for mmd_rbf)
numpy                                     → We have 2.5.2
```

**Status:** FedCC **cannot run unmodified** on this machine due to `cuda:0` hardcoding.  
**Decision:** We will NOT modify FedCC source files. Our FedCC baseline will be a clean reimplementation that follows the FedCC algorithm but is device-agnostic.

---

## 6. Version Compatibility Notes

### PyTorch 2.x vs 1.x Changes (Relevant to Reference Code)

| Old API | New API | Impact |
|---|---|---|
| `torch.tensor(...).cuda()` | `.to(device)` or `.to('mps')` | FedCC uses old API |
| `torch.backends.cudnn.benchmark` | Same | FedCVG may use this; harmless if CUDA not present |
| `KMeans(n_init=10)` warning | Default changed in sklearn 1.2+ | Pass `n_init=10` explicitly |
| `np.bool` (removed NumPy 1.24+) | `bool` | Check if any reference code uses `np.bool` |

### Python 3.14 Specific Notes

- Python 3.14 is newer than what FedCVG/FedCC targeted (Python 3.6–3.9)
- Key changes: `typing` module improvements, no breaking changes for our use case
- `f-string` improvements are backward compatible

---

## 7. Verification Commands

Run these to verify the environment is correctly set up:

```bash
source .venv/bin/activate

# Verify PyTorch and device
python -c "import torch; print(torch.__version__, 'MPS:', torch.backends.mps.is_available())"

# Verify all packages
python -c "import sklearn, numpy, scipy, pandas, matplotlib, seaborn, yaml, tqdm; print('All packages OK')"

# Run unit tests
python -m pytest tests/ -v
```

---

## 8. Change Log

| Date | Change | Reason |
|---|---|---|
| 2026-09-30 | Initial environment setup | First milestone |
| 2026-09-30 | Chose MPS over CPU as primary device | Apple Silicon available; MPS confirmed working |
| 2026-09-30 | Decided NOT to modify FedCC cuda:0 references | Research integrity — external code must remain unmodified |

---

*Last updated: 2026-09-30*
