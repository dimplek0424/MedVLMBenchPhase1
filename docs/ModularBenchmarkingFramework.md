## 🧩 Modular Benchmarking Framework

MedVLM Bench is designed as a modular, extensible research toolkit.  
Each model or task fits into one of the following components:

### 📁 1. Configurations (`configs/`)
YAML files defining:
- dataset paths
- preprocessing settings
- model hyperparameters
- evaluation parameters

### 📁 2. Scripts (`scripts/`)
Python scripts implementing:
- embedding extraction
- retrieval evaluation
- zero-shot classification
- generation pipelines (Phase-2)

### 📁 3. Notebooks (`notebooks/`)
Used for:
- exploratory analysis (EDA)
- prototype experiments
- visualization and debugging

### 📁 4. Reports (`reports_phase1/`, `EDA/`)
Stores:
- metric tables
- generated outputs
- EDA PDFs
- qualitative evaluations

### ➕ Adding New Tasks
New modules (e.g., segmentation, RadGraph extraction, fine-grained classification) can be added by:
- creating a config file
- writing a corresponding script
- optionally adding a visualization notebook

This ensures clean, reproducible expansion across phases.
