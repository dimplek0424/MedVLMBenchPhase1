# Towards Comprehensive Benchmarking of Medical Vision–Language Models (VLMs)  
### MedVLM Phase 1  

---

## Overview
This repository supports the **first phase** of an ongoing research project on **efficient Medical Vision–Language Models (VLMs)** for radiology applications.  
The goal is to establish **reproducible baselines** for *small and efficient* models on key imaging tasks — **zero-shot classification**, **multimodal retrieval**, and **report summarization** — using publicly available datasets.

This work is conducted under the **mentorship of Dr. Sanjan T. P. Gupta** (AI for Healthcare) and has been **recognized with:**
- 🧾 **Poster Talk** — *GIW XXXIV ISCB Main Conference 2025*  
- 🎤 **Oral Talk** — *ASCS 2025 Symposium on Advanced Computing & Systems*  

---

## Research Focus
Large-scale multimodal models (e.g., GPT-4V, CLIP, LLaVA-Med) deliver excellent results but are challenging to deploy in healthcare due to compute, interpretability, and data-governance limitations.  
This research benchmarks **smaller, domain-specific medical VLMs (< 10 B parameters)** to understand **accuracy–efficiency trade-offs** in clinical AI.

**Models currently explored**
- [MedCLIP](https://github.com/UCSD-AI4H/MedCLIP) – Contrastive learning for image–text alignment  
- [BioMedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) – PubMedBERT-based cross-modal encoder  
- [CheXzero](https://github.com/rajpurkarlab/chexzero) – Zero-shot chest X-ray classification  
- [LLaVA-Med](https://github.com/microsoft/LLaVA-Med) / [XrayGPT](https://github.com/UCSD-AI4H/XrayGPT) – Vision–language reasoning and report generation  

---

## Dataset Usage

### **Phase 1 — Local experiments**  
For initial benchmarking and reproducibility, we use the **Indiana University Chest X-Ray dataset**:  
🔗 [Kaggle – Chest X-rays (Indiana University)](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)

This dataset is compact, publicly available, and manageable on local systems — ideal for CPU-based experimentation without institutional restrictions.

### **Phase 2 — Clinical-scale expansion**  
Next, the pipeline will extend to **MIMIC-CXR v2.1.0** after setting up the Kaggle GPU utilization workflow.  
🔗 [PhysioNet – MIMIC-CXR Dataset](https://physionet.org/content/mimic-cxr/2.1.0/)

> ⚠️ **No patient-identifiable information is stored or shared.**  
> All experiments rely on public datasets and open-source models only.

---

## Setup Instructions

### **1 — Clone the repository**
```bash
git clone https://github.com/dimplek0424/MedVLMPhase1.git
cd MedVLMPhase1
```

---

### **2 — Create the Environment**
```bash
conda create -n medvlm python=3.9 -y
conda activate medvlm
```

---

### **3 — Install Dependencies**
```bash
pip install -r requirements.txt
```

---

### **4 — Directory Structure**

```
MedVLMPhase1/
│
├── notebooks/
│   ├── medclip_demo.py           # MedCLIP baseline (CPU-friendly)
│   ├── biomedclip_local.py       # BioMedCLIP evaluation script
│   └── evaluate_vlm_models.py    # Unified benchmarking & metrics
│
├── data/        # Local dataset (excluded from Git)
├── outputs/     # Metrics & logs (excluded)
├── requirements.txt
└── README.md
```

---

## Evaluation Tasks

| **Task** | **Description** | **Metrics** |
|:--|:--|:--|
| **Zero-Shot Classification** | Predict radiological findings using prompt templates without fine-tuning | Accuracy · F1 · AUC |
| **Cross-Modal Retrieval** | Retrieve the most relevant report given an image (and vice versa) | Recall@K · Cosine Similarity |
| **Report Summarization** | Generate concise, clinical-style summaries | BLEU · ROUGE-L · BERTScore |
| **Efficiency Analysis** | Quantify model performance on CPU | Inference Time · Memory · FLOPs |

---

## Model Parameter Overview

| **Model** | **Architecture** | **Parameters** | **Core Capability** |
|:--|:--|:--|:--|
| **MedCLIP** | ViT-Base + BioClinicalBERT | ≈ 86 M | Image–text alignment |
| **BioMedCLIP** | ViT-Base + PubMedBERT | ≈ 120 M | Cross-modal retrieval |
| **CheXzero** | ResNet-50 + Domain LM | ≈ 90 M | Zero-shot classification |
| **LLaVA-Med / XrayGPT** | Vision encoder + LLM decoder | 7 B + | Report reasoning / summarization |

---

## Evaluation Pipeline

- 1️⃣  Preprocessing — Resize & normalize images (224 × 224 px)
- 2️⃣  Feature Extraction — Generate embeddings via MedCLIP / BioMedCLIP
- 3️⃣  Zero-Shot or Retrieval Evaluation — Compute similarity or class predictions
- 4️⃣  Summarization Phase — Use LLaVA-Med / XrayGPT for report generation
- 5️⃣  Efficiency Metrics — Record latency, memory usage & throughput

---

## Next Steps

- **Extend benchmarking** to MIMIC-CXR Phase 2 (post-access approval)  
- **Apply quantization** and **LoRA fine-tuning** for efficient inference  

---

## Ethics & Compliance

This repository adheres to ethical AI research and data-governance standards:

- Uses only **public, de-identified datasets** (Indiana University Chest X-rays)  
- Complies with **PhysioNet Data Use Agreement (DUA)** for MIMIC-CXR access  
- Employs only **open-source pretrained models** under their respective licenses  
- No patient or personally identifiable information (PII) is stored or shared  

---

## References

1. **Wang Z. et al.** *MedCLIP: Contrastive Learning for Medical Vision–Language Understanding.*  
   arXiv preprint, 2023. [🔗 arXiv:2303.XXXX](https://arxiv.org/abs/2303.XXXX)

2. **Microsoft Research.** *BioMedCLIP: Cross-Modal Pretraining for Biomedical Understanding.*  
   arXiv preprint, 2023. [🔗 arXiv:2301.XXXX](https://arxiv.org/abs/2301.XXXX)

3. **Tiu E. et al.** *CheXzero: Training Medical AI Models Without Labels.*  
   *Nature Medicine*, 2022. [🔗 DOI:10.1038/s41591-022-02157-4](https://doi.org/10.1038/s41591-022-02157-4)

4. **Li Y. et al.** *LLaVA-Med: Large Language-and-Vision Assistant for Biomedicine.*  
   arXiv preprint, 2024. [🔗 arXiv:2401.XXXX](https://arxiv.org/abs/2401.XXXX)

---

## License

This project is released under the **MIT License** for research and educational purposes.  
If you build upon or reproduce this work, please provide proper attribution.

📄 *License text available in* [`LICENSE`](LICENSE)

---

## Learn More

For extended methodology, dataset notes, and evaluation design:  
📖 Read the detailed [**PROJECT_OVERVIEW.md**](PROJECT_OVERVIEW.md)

> Includes dataset registry, preprocessing flow, model configuration, and evaluation metrics.
