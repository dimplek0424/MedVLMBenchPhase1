# EXACT IMPLEMENTATION OF MEDCLIP README EXAMPLES
# ------------------------------------------------
# This script reproduces BOTH README demos:
# 1. CLIP-style similarity with text + example image
# 2. Prompt-based disease classification

import torch
from PIL import Image

from medclip import (
    MedCLIPModel,
    MedCLIPVisionModelViT,
    MedCLIPProcessor,
    PromptClassifier
)
from medclip.prompts import generate_chexpert_class_prompts, process_class_prompts


# ------------------------------------------------
# DEVICE SETUP
# ------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}\n")


# ------------------------------------------------
# PART 1 — CLIP-style similarity (README section 1)
# ------------------------------------------------
print("=== PART 1: CLIP-style similarity ===")

# Prepare processor + image
processor = MedCLIPProcessor()
image = Image.open('./example_data/view1_frontal.jpg')

inputs = processor(
    text=[
        "lungs remain severely hyperinflated with upper lobe emphysema",
        "opacity left costophrenic angle is new since prior exam ___ represent some loculated fluid cavitation unlikely"
    ],
    images=image,
    return_tensors="pt",
    padding=True
)

# Send tensors to device
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        inputs[k] = v.to(device)

# Load model as in README
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()
model = model.to(device)

outputs = model(**inputs)

print("Output keys:", outputs.keys())  # should match README
# Expected: dict_keys(['img_embeds', 'text_embeds', 'logits', 'loss_value', 'logits_per_text'])


# ------------------------------------------------
# PART 2 — Prompt-based disease classification (README section 2)
# ------------------------------------------------
print("\n=== PART 2: Prompt-based Disease Classification ===")

processor2 = MedCLIPProcessor()

model2 = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model2.from_pretrained()
model2 = model2.to(device)

clf = PromptClassifier(model2, ensemble=True)
clf = clf.to(device)

# Load image again
image2 = Image.open('./example_data/view1_frontal.jpg')
inputs2 = processor2(images=image2, return_tensors="pt")

for k, v in inputs2.items():
    if isinstance(v, torch.Tensor):
        inputs2[k] = v.to(device)

# Generate CheXpert disease prompts exactly like README
cls_prompts = process_class_prompts(generate_chexpert_class_prompts(n=10))
inputs2["prompt_inputs"] = cls_prompts

# Make prediction
output = clf(**inputs2)
print(output)
# Expected shape & structure:
# {
#   'logits': tensor([[...]]),
#   'class_names': ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
# }

print("\n=== README Demo Completed Successfully ===\n")
