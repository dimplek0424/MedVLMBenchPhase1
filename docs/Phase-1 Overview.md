## 🧪 Phase-1 Overview — IU Chest X-ray Baseline Benchmark

Phase-1 establishes a reproducible evaluation pipeline using the **Indiana University Chest X-ray dataset (OpenI)** — a compact, fully public dataset ideal for rapid experimentation.

### 🔍 Objectives
- Build a CPU-friendly reproducible baseline framework.
- Evaluate small medical VLMs without relying on large GPUs.
- Standardize preprocessing, embeddings, and ranking metrics.
- Generate foundational metrics for comparison in later phases.

### 📦 Models Implemented in Phase-1
- **MedCLIP** — contrastive image–text alignment  
- **BioMedCLIP** — biomedical vision-language pretraining  
- **CheXzero** — zero-shot radiology classification  

### 🧪 Tasks Implemented
- Image embedding extraction  
- Report/text embedding extraction  
- Image↔report retrieval (cosine similarity ranking)  
- Zero-shot pathology classification (CheXpert-lite labels)  
- Baseline visualization + EDA  

### 📤 Outputs
- Preprocessed IU-Xray dataset splits  
- Baseline retrieval/zero-shot performance  
- Global EDA reports (PDFs + notebooks)  
- Kaggle GPU workflow validated  
- Reference metrics for Phase-2 scaling  
