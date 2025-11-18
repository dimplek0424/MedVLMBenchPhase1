# Towards Comprehensive Benchmarking of Medical Vision–Language Models (Med‑VLMs)

**A Unified Research Framework for Efficient, Trustworthy, and Clinically Deployable Medical Vision–Language Models**

Medical imaging workflows rely on the integration of **radiology images** and **free‑text reports**. While Large Vision–Language Models (LVLMs) such as GPT‑4V and LLaVA‑Med demonstrate strong medical reasoning, they remain challenging to deploy in real clinical environments due to:
- heavy computational requirements,
- privacy and data‑governance barriers,
- limited interpretability,
- reliance on cloud‑scale infrastructure.

This motivates a systematic study of **small and domain‑specific models (<10B parameters)**—including MedCLIP, BioMedCLIP, CheXzero, MedFILIP, MedBridge, and radiology‑specific SLMs—which offer:
- lower latency,
- reduced VRAM requirements,
- improved transparency,
- on‑premise feasibility for hospitals.

This repository provides the **benchmarking foundation** for the manuscript:  
📄 *"Towards Comprehensive Benchmarking of Medical Vision Language Models"* fileciteturn0file0

It aims to serve as a **research‑first, reproducible benchmark suite** for evaluating efficiency, accuracy, trustworthiness, and clinical readiness of Med‑VLMs.

---

# 🔭 High‑Level Research Overview
This project investigates three pillars of Med‑VLM performance:

### **1. Task Performance**
- Zero‑shot classification (CheXpert labels)
- Multimodal retrieval (image–report / report–image)
- Report summarization and impression generation

### **2. Efficiency & Deployability**
- latency and throughput
- VRAM / CPU footprint
- model size, FLOPs, quantization behavior
- stability across seeds

### **3. Trustworthiness & Reliability**
- factual correctness
- calibration error
- robustness to perturbations
- rare‑finding performance

This unified framework will later extend beyond chest X‑rays to CT/MRI/ophthalmology datasets.

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
Formal citations and expanded methodology appear in the draft manuscript:  
📄 *"Towards Comprehensive Benchmarking of Medical Vision Language Models"* fileciteturn0file0

---

# 👩‍💻 Maintainer
**Dimple Khatri** — AI for Healthcare Researcher  
Contact: dimplek0424@gmail.com
