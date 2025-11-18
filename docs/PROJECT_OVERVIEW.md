# 🩺 Project Overview — MedVLM Phase 1  
**Title:** Towards Comprehensive Benchmarking of Medical Vision–Language Models (VLMs)  
**Principal Investigator:** Dimple Khatri  
**Mentor:** Dr. Sanjan T. P. Gupta (AI for Healthcare)  
**Current Phase:** Local benchmarking using Indiana University Chest X-Ray dataset  
**Planned Conference Submission:** ISCB Asia 2025  

---

## 1️⃣ Research Objectives
- Build reproducible baselines for **medical VLMs** (e.g., MedCLIP, BioMedCLIP, CheXzero).  
- Evaluate models for **zero-shot classification**, **multimodal retrieval**, and **report summarization**.  
- Study **accuracy vs. efficiency** trade-offs on CPU-only setups.  
- Design scalable workflows for eventual integration with **MIMIC-CXR** under data-use compliance.  

---

## 2️⃣ Project Phases

| Phase | Dataset | Purpose | Notes |
|:------|:---------|:--------|:------|
| **Phase 1** | [Indiana University Chest X-Ray (Open-i / Kaggle)](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university) | Lightweight local benchmarking | Enables experimentation without institutional restrictions |
| **Phase 2** | [MIMIC-CXR v2.1.0 (PhysioNet)](https://physionet.org/content/mimic-cxr/2.1.0/) | Scalable benchmarking with real clinical data | Requires DUAs + ethics approval |
| **Phase 3** | Quantized + Explainable VLMs | Efficiency & interpretability | Integration with Grad-CAM / LoRA / attention maps |

---

## 3️⃣ Repository Structure
MedVLMPhase1/
├── data/ # Local dataset storage (excluded)
├── notebooks/
│ ├── medclip_demo.py → CPU-safe MedCLIP baseline
│ ├── biomedclip_local.py → BioMedCLIP evaluation
│ ├── evaluate_vlm_models.py → Task-wise comparison script
│
├── outputs/ # Model embeddings, metrics (excluded)
├── requirements.txt
├── README.md
└── PROJECT_OVERVIEW.md # ← this file

---

## 4️⃣ Model Registry and Links

| Model | Paper / Repo | Domain | Key Capability |
|:------|:--------------|:-------|:---------------|
| **MedCLIP** | [GitHub](https://github.com/UCSD-AI4H/MedCLIP) · [arXiv 2023](https://arxiv.org/abs/2301.01558) | Radiology | Image–text alignment |
| **BioMedCLIP** | [Hugging Face](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | Biomedical | Cross-modal retrieval |
| **CheXzero** | [GitHub](https://github.com/rajpurkarlab/chexzero) · [Nature Med 2022](https://www.nature.com/articles/s41591-022-01920-1) | Chest X-Ray | Zero-shot classification |
| **LLaVA-Med** | [GitHub](https://github.com/microsoft/LLaVA-Med) · [arXiv 2024](https://arxiv.org/abs/2403.18798) | Biomedical | Vision-language reasoning |
| **XrayGPT** | [GitHub](https://github.com/UCSD-AI4H/XrayGPT) | Radiology | Report summarization + QA |

---

## 5️⃣ Evaluation Tasks and Metrics
| Task | Metrics | Description |
|:------|:---------|:-------------|
| **Zero-Shot Classification** | Accuracy, F1, AUC | Label prediction without fine-tuning |
| **Retrieval (Img ↔ Text)** | Recall@K, Cosine Sim | Cross-modal embedding alignment |
| **Report Summarization** | BLEU, ROUGE-L, BERTScore | Textual coherence & semantic fidelity |
| **Efficiency Analysis** | Inference Time, Memory Use | Resource-aware evaluation on CPU |

---

## 6️⃣ Ethical and Data Governance
- Only **publicly available datasets** are used in this phase.  
- No PHI or identifiable clinical data are stored or shared.  
- All models and data follow their respective **licenses and DUAs**.  

---

## 7️⃣ Next Steps
- Deploy MedCLIP/BioMedCLIP on full MIMIC-CXR dataset (Phase 2).  
- Introduce **LoRA fine-tuning** and **quantization**.  
- Extend benchmarks to **XrayGPT / LLaVA-Med** for report generation.  
- Submit full paper to ISCB Asia 2025 or alternate venue.  

---

## 🧾 License
This repository is open-source for research and educational use under the **MIT License**.  
Please cite appropriately when using any component.
