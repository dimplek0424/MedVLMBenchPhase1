## 🧬 Model Landscape: Implemented, In-Progress, and Future Scope

This benchmark covers a spectrum of medical VLMs, categorized by their integration stage in the project.

| Model        | Year | Dataset(s) Used | Key Method / Architecture               | Primary Tasks                           | Status               |
|--------------|------|----------------|-------------------------------------------|------------------------------------------|----------------------|
| **MedCLIP**  | 2022 | MIMIC-CXR       | Contrastive pretraining (CLIP-style)      | Retrieval, zero-shot classification      | **Implemented (Phase-1)** |
| **BioMedCLIP** | 2023 | PMC + MIMIC    | Biomedical vision–language pretraining    | Retrieval, classification                 | **Implemented (Phase-1)** |
| **CheXzero** | 2023 | MIMIC-CXR       | Supervised contrastive learning           | Zero-shot classification                 | **InProgress (Phase-1)** |
| **LLaVA-Med** | 2024 | MIMIC-CXR, OpenI | Vision encoder + LLM alignment           | Report generation, QA                    | **Planned (Phase-2)** |
| **XrayGPT**  | 2024 | MIMIC-CXR       | MedCLIP encoder + GPT-style decoder       | Summarization, QA                        | **Planned (Phase-2)** |
| **MedFILIP** | 2025 | MIMIC-CXR, CheXpert | Triplet supervision                     | Fine-grained disease detection           | **Future Scope**     |
| **MedBridge**| 2025 | MIMIC-CXR       | Frozen encoders + adapter-based fusion    | Diagnosis, benchmarking                   | **Future Scope**     |

This staged structure reflects the project’s evolution from **Phase-1 baselines → Phase-2 expansion → long-term generalization**.
