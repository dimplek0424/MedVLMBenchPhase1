## 📊 Evaluation Tasks

This benchmark evaluates Med-VLMs across diverse tasks reflecting real radiology workflows.

### 🔹 Zero-Shot Classification
Predict CheXpert-style labels without fine-tuning.
**Metrics**: AUC, Accuracy, F1, Precision-Recall.

### 🔹 Multimodal Retrieval
Retrieve the relevant report given an image, or vice-versa.
**Metrics**: Recall@K, Mean Reciprocal Rank (MRR), Cosine Similarity.

### 🔹 Report Summarization / Impression Generation
Generate clinical-style summaries from chest X-rays.
**Metrics**: ROUGE-L, BLEU-4, BERTScore, RadGraph-F1 (optional).

### 🔹 Representation Quality
Evaluate embedding separability and alignment.
**Metrics**: t-SNE/UMAP clusters, intra-cluster variance, similarity matrices.

### 🔹 Efficiency & Deployability
Measure inference speed and hardware resource usage.
**Metrics**: latency, VRAM/CPU footprint, FLOPs, throughput, quantization behavior.
