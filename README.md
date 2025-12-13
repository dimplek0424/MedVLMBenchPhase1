# Towards Comprehensive Benchmarking of Medical Vision–Language Models (Med-VLMs)

**A Unified Research Framework for Efficient, Trustworthy, and Clinically Deployable Medical Vision–Language Models**

Medical imaging workflows depend on the integration of **radiology images** and **clinical free-text reports**. While Large Vision–Language Models (LVLMs) such as GPT-4V and LLaVA-Med demonstrate impressive medical reasoning capabilities, they remain difficult to deploy in real clinical environments due to:

- high computational and memory requirements,  
- strict privacy and data-governance constraints,  
- limited interpretability and controllability,  
- reliance on cloud-scale, non-local infrastructure.

These constraints motivate a systematic evaluation of **small and domain-specialized Medical Vision–Language Models (<10B parameters)**—such as **MedCLIP**, **BioMedCLIP**, **CheXzero**, **LLaVA-Med**, **XrayGPT**, **MedFILIP**, **MedBridge**, and radiology-specific SLMs—which offer:

- substantially lower latency and memory footprint,  
- improved transparency and reliability,  
- feasibility for on-premise hospital deployment,  
- stronger alignment with radiology-domain supervision.

This repository serves as a **research-first, reproducible benchmark suite** designed to evaluate the **efficiency, accuracy, robustness, and clinical readiness** of small and mid-sized Med-VLMs across multiple datasets and tasks.

---

### 📄 Publication

**Dimple Khatri and Sanjan TP Gupta**  
*Towards Comprehensive Benchmarking of Medical Vision-Language Models.*   
Abstract published in **Briefings in Bioinformatics** (Oxford Academic),  
[Volume 26, Supplement 1 — GIW XXXIV ISCB-Asia 2025 Abstract Book](https://academic.oup.com/bib/issue/26/Supplement_1), 2025.

---

### 🏅 Mentorship & Scientific Recognition

This research is conducted under the mentorship of **Dr. Sanjan T. P. Gupta (AI for Healthcare)**.

The work has been formally recognized through peer-reviewed international venues:
- 🧾 **Poster Presentation — GIW XXXIV ISCB-Asia 2025 (Main Conference)**
- 🎤 **Oral Presentation — ASCS 2025 (ISCB Student Council Symposium)**

These recognitions reflect the scientific rigor and relevance of this multi-phase benchmarking study on medical vision–language models.

---

## 🔭 High-Level Research Overview

This benchmarking framework evaluates Medical Vision–Language Models across three foundational dimensions:

### 1. Core Task Performance
Med-VLMs are assessed on radiologically meaningful downstream tasks, including:
- **Zero-shot classification** (CheXpert-derived findings)
- **Multimodal retrieval** (image→report and report→image search)
- **Clinical impression summarization** using VLM decoders
- **Representation quality** through embedding alignment and ranking metrics

These tasks reflect the modalities most relevant to radiology workflows: visual interpretation, textual grounding, and summarization.

---

### 2. Efficiency & Clinical Deployability
To support real-world healthcare deployment, the benchmark measures:
- **Latency and throughput** (CPU and GPU)
- **VRAM/CPU memory footprint**
- **Model size and FLOPs**
- **Quantization behavior** (FP16, INT8, INT4)
- **Stability across random seeds and patient-level splits**

This enables fair comparison between models that vary widely in scale—from ~80M parameters to multi-billion-parameter VLMs.

---

### 3. Trustworthiness & Reliability
Med-VLMs must not only be accurate but also *clinically reliable*.  
We evaluate:
- **Factual correctness** of generated summaries
- **Calibration error** (ECE/MCE)
- **Robustness** under perturbations and image quality degradation
- **Rare-finding performance** in low-prevalence conditions
- **Failure mode characterization** through error distribution analysis

This reflects growing emphasis on safety-critical AI in medical imaging.

---

Together, these components form a unified benchmark for understanding how small and mid-sized medical VLMs perform across **accuracy, efficiency, and trustworthiness** — the three pillars required for practical clinical integration.


---

# 🧭 Project Phases
The repository is organized around the evolution of the research pipeline.

---

# ## **Phase 1 — Establishing Baseline Benchmarks (IU Chest X‑ray)**
### **Goal:** Build a reproducible, CPU‑friendly baseline pipeline using publicly available data.

Phase 1 focuses on the **Indiana University Chest X‑ray dataset**, chosen because it is:
- fully public and de‑identified,
- small enough for rapid iteration,
- paired with high‑quality radiology reports,
- ideal for early CPU‑level prototyping.

### **Models evaluated in Phase 1:**
- **MedCLIP** — contrastive image–text alignment
- **BioMedCLIP** — vision encoder + PubMedBERT
- **CheXzero** — zero‑shot classification

### **Tasks implemented:**
- image embedding extraction
- text embedding extraction
- cosine‑similarity retrieval
- top‑K ranking
- zero‑shot pathology classification

### **Outputs from Phase 1:**
- IU‑Xray preprocessing and splits
- EDA notebooks + PDF reports
- end‑to‑end Kaggle GPU workflows
- reproducible MedCLIP/BioMedCLIP benchmarks
- baseline metrics for all Phase‑2 comparisons

This phase forms the foundation for scaling up to clinical datasets.

---

# ## **Phase 2 — Scaling to CheXpert and MIMIC‑CXR (Ongoing)**
### **Goal:** Build a comprehensive, clinically meaningful benchmark that assesses accuracy, efficiency, and trust.

Phase 2 expands the pipeline to:
- **CheXpert** (label‑rich, high‑quality dataset)
- **MIMIC‑CXR v2.1.0** (largest public CXR dataset)

### **New components introduced:**
#### **1. Advanced EDA (global)**
- label frequency & imbalance
- metadata and device analysis
- density/intensity distributions

#### **2. Larger model families**
- LLaVA‑Med
- XrayGPT
- MedBridge
- MedFILIP
- SLM baselines (BioClinicalBERT, TinyBERT, DistilBERT)

#### **3. Efficiency‑focused experiments**
- 8‑bit / 4‑bit quantization
- qLoRA fine‑tuning
- throughput + latency benchmarking
- VRAM footprint tracking

#### **4. Trustworthiness assessment**
- factual alignment
- calibration metrics
- robustness to perturbations
- rare‑finding performance

### **Expected Phase‑2 Outputs:**
- unified metrics tables (AUC, ROUGE‑L, Recall@K)
- cross‑dataset evaluation
- quantization & PEFT ablation studies
- trustworthiness report
- comparison across model architectures

Phase 2 will form the basis for the main results in the final paper.

---

# ## **Initial POC Experiments (Local Prototyping)
Before formalizing Phase 1, small exploratory experiments were run locally to:
- validate loaders,
- test preprocessing variations (PIL vs OpenCV),
- run mini retrieval experiments,
- build a first prototype for the MedCLIP/BioMedCLIP pipeline,
- verify Kaggle GPU compatibility.

These experiments informed the more structured pipelines found in Phase 1.

---

# 📐 Model Parameter Comparison (Current Baselines)
| Model | Architecture | Parameters | Core Capability |
|-------|-------------|------------|-----------------|
| **MedCLIP** | ViT-Base + BioClinicalBERT | ≈ 86M | Image–text alignment |
| **BioMedCLIP** | ViT-Base + PubMedBERT | ≈ 120M | Cross-modal retrieval |
| **CheXzero** | ResNet-50 + Domain LM | ≈ 90M | Zero-shot classification |
| **LLaVA-Med / XrayGPT** | Vision encoder + LLM decoder | 7B+ | Report reasoning & summarization |

This comparison highlights the accuracy–efficiency trade-offs motivating our focus on **small, deployable Med-VLMs**.

---

# 🧩 Modular Benchmarking Framework
MedVLM Bench is designed as a **modular, extensible research toolkit**.

Each baseline model has:
- **Config files** in `configs/`
- **Dedicated scripts** in `scripts/` or phase-specific `notebooks/`
- **Metrics & outputs** tracked in `reports_phase1/`, `EDA/`, or model-specific outputs

### Adding New Tasks
New tasks (e.g., projection learning, disease-label extensions, advanced retrieval, RadGraph entity extraction) can be added as plug-in modules following the structure of existing scripts such as:
- `medclip_demo.py`
- `projection_medclip.py`

This modular design supports Phase 2 expansion and future multi-dataset evaluation.

---

---

# 🗂 Repository Organization
```
MedVLMBench/
│
├── data/                      # Local datasets (ignored in Git)
│
├── EDA/                       # Global EDA notebooks + PDF reports
│   ├── notebooks_eda/
│   └── eda_reports/
│
├── docs/                      # Workflow docs + project overview
│
├── phase1/                    # IU-Xray baseline pipeline
│   ├── configs/
│   ├── notebooks/
│   ├── scripts/
│   ├── reports_phase1/
│   └── medvlm_core/
│
├── phase2/                    # CheXpert + MIMIC-CXR benchmark (in progress)
│   └── ...
│
├── requirements.txt
├── requirements_kaggle.txt
└── README.md
```

---

# ⚙️ Setup Instructions
### **1 — Clone the repository**
```bash
git clone https://github.com/dimplek0424/MedVLMBenchPhase1.git
cd MedVLMBenchPhase1
```

### **2 — Create conda environment**
```bash
conda create -n medvlm python=3.10 -y
conda activate medvlm
```

### **3 — Install dependencies**
```bash
pip install -r requirements.txt
```
For Kaggle:
```bash
pip install -r requirements_kaggle.txt
```

---

# 📊 Evaluation Tasks & Metrics
| Task | Description | Metrics |
|------|-------------|---------|
| Zero‑Shot Classification | Predict CheXpert labels | AUC, F1, Accuracy |
| Cross‑Modal Retrieval | Image ↔ Report search | Recall@K, Cosine Similarity |
| Report Summarization | Generate clinical impressions | ROUGE‑L, BLEU, BERTScore |
| Efficiency Analysis | Measure deployability | VRAM, latency, FLOPs |

---

# 🛡 Ethics & Compliance
- Only uses public, de‑identified datasets
- Complies with PhysioNet DUA
- No PHI or sensitive information stored
- All models follow their original licenses

---

# 📚 References

### Vision–Language Models
- Wang Z. et al. **“MedCLIP: Contrastive Learning from Unpaired Medical Images and Text.”** arXiv:2301.08147 (2023).
- Zhang Y. et al. **“BioMedCLIP: A Vision-Language Foundation Model for the Biomedical Domain.”** arXiv:2303.00915 (2023).
- Tiu E. et al. **“Expert-level detection of pathologies from unlabelled chest X-ray images.”** Nature Biomedical Engineering (2022).
- Li Y. et al. **“LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine.”** arXiv:2401.02413 (2024).

### Datasets
- Demner-Fushman D. et al. **“Preparing a collection of radiology examinations for distribution and retrieval.”** JAMIA (2012). *(Indiana University Chest X-ray Dataset)*
- Irvin J. et al. **“CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.”** AAAI 2019; arXiv:1901.07031.
- Johnson A. et al. **“MIMIC-CXR: A large publicly available database of labeled chest radiographs.”** arXiv:1901.07042 (2019).

### Methods
- Radford A. et al. **“Learning Transferable Visual Models From Natural Language Supervision.”** ICML 2021; arXiv:2103.00020. *(CLIP framework)*

---

# 👩‍💻 Maintainer
**Dimple Khatri** — AI for Healthcare Researcher  
Contact: dimplek0424@gmail.com

---

### Citation

If you use this work, please cite:

> Dimple Khatri and Sanjan TP Gupta (2025).  
> *Towards comprehensive benchmarking of medical vision-language models.*  
> **Briefings in Bioinformatics**, 26(Suppl. 1). Oxford Academic..
