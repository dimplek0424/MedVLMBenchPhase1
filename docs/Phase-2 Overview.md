## 🚀 Phase-2 Overview — CheXpert and MIMIC-CXR Expansion

Phase-2 scales the benchmark to clinically meaningful datasets and introduces advanced VLM families and trustworthiness metrics.

### 🔍 Objectives
- Extend evaluation to **CheXpert** and **MIMIC-CXR v2.1.0**.
- Incorporate VLMs capable of generation and reasoning.
- Benchmark efficiency/latency at clinical scale.
- Add trustworthiness evaluations (factuality, robustness).

### 🧬 Phase-2 Model Additions
- **LLaVA-Med** — multimodal instruction-aligned VLM  
- **XrayGPT** — MedCLIP encoder + GPT decoder  
- (Optional future models) MedFILIP, MedBridge  

### 📊 Expanded Metrics
- Report generation quality (ROUGE-L, BLEU, BERTScore)  
- Calibration error (ECE/MCE)  
- Robustness to perturbations  
- Rare-finding detection performance  
- Efficiency (FP16/INT8/INT4 quantization)  

### 📦 Deliverables
- CheXpert preprocessing pipeline  
- MIMIC-CXR benchmark suite  
- Unified evaluation scripts across models  
- Trustworthiness analysis report  
- Model comparison tables and ablations  
