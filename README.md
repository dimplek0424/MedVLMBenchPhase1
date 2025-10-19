# MedVLM Phase 1

Local benchmark of medical vision-language models (CPU-only).

### Models
- **MedCLIP** – zero-shot classification  
- **BiomedCLIP** – zero-shot classification  
- Unified evaluation comparing both.

### Folder structure
- data/ # (ignored) datasets such as IU CXR, MIMIC-CXR
- medclip-vit-base-patch16/ # (ignored) model weights / cache
- notebooks/ # runnable Python scripts
- outputs/ # (ignored) generated predictions & figure


### Quick start
```bat
conda create -n medvlm python=3.10 -y
conda activate medvlm
pip install -r requirements.txt

python notebooks\medclip_demo.py
python notebooks\biomedclip_local.py
