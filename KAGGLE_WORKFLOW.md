# 🧪 Kaggle GPU Workflow — MedVLMBench Phase 1

> **Goal:**  
> Run the **MedCLIP projection benchmark** on Kaggle GPU (Tesla P100) using the **Indiana University Chest X-ray** dataset, then generate evaluation charts and calibration metrics.

---

## 📂 Repository & Key Paths

**Public Repo:** [`MedVLMBenchPhase1`](https://github.com/dimplek0424/MedVLMBenchPhase1)  
**Kaggle Notebook:** `notebooks/MedCLIP_KaggleRunner_MedVLMPhase1.ipynb`

| Component | Path | Purpose |
|------------|------|---------|
| **Main Script** | `scripts/projection_medclip.py` | Runs the MedCLIP projection benchmark |
| **Evaluation** | `scripts/evaluate_views.py` | Generates confusion matrix + charts |
| **Dataset Config** | `configs/dataset_iu_v03_full.yaml` | IU-CXR dataset definition |
| **Task Config** | `configs/task_projection_v01.yaml` | Task labels (Frontal vs Lateral) |
| **Kaggle Overrides** | `configs/dataset_iu_v03_full.kaggle.yaml` | Uses `DATA_DIR` / `OUTPUT_DIR` envs |
| **Helpers** | `scripts/print_paths.py`, `medvlm_core/io.py`, `medvlm_core/dataloader.py` | Centralized path and loader logic |

---

## ⚙️ Why These Changes Matter

- **Env-Aware Config** → same YAML works locally and on Kaggle by setting `DATA_DIR` and `OUTPUT_DIR`.  
- **Centralized I/O** → `get_dataset_paths()` ensures consistent path resolution.  
- **Loader Wrapper** → `make_loader_from_cfg(cfg)` replaces hard-coded paths.  
- **Safe MedCLIP Install** → uses `--no-deps` and manual `wget textaugment` add-ons to avoid downgrades.  
- **Tolerant Model Loader** → avoids crashes from extra keys in newer MedCLIP checkpoints.

---

## 🚀 Step-by-Step (Fresh Kaggle Session)

> **⚠️ Note:** Every GPU reset or session restart clears the VM. Re-run all setup cells before running scripts.

### 1️⃣ Start Notebook (GPU On + Dataset Attached)

- **Accelerator:** GPU → Tesla P100 (GPU P100)
- **Add Data:** `chest-xrays-indiana-university`

---

### 2️⃣ Clone Repo & Set Environment

```bash
!git clone https://github.com/dimplek0424/MedVLMBenchPhase1.git
%cd /kaggle/working/MedVLMBenchPhase1

import os
os.environ["DATA_DIR"]   = "/kaggle/input/chest-xrays-indiana-university"
os.environ["OUTPUT_DIR"] = "/kaggle/working/outputs"

!python scripts/print_paths.py
```

---

### 3️⃣ Install Dependencies (Safe for Kaggle)

```bash
# Base requirements
!pip install -q -r requirements_kaggle.txt

# MedCLIP without pinned older deps
!pip install -q --no-deps git+https://github.com/RyanWangZf/MedCLIP.git@main

# Minimal extras MedCLIP expects
!pip install -q wget textaugment
```
