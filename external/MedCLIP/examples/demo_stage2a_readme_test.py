from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
from medclip.prompts import generate_chexpert_class_prompts, process_class_prompts
from medclip import PromptClassifier
from PIL import Image
import torch

def main():
    # 1. Load processor + sample image
    processor = MedCLIPProcessor()
    image_path = "./example_data/view1_frontal.jpg"
    image = Image.open(image_path)

    print(f"Loaded example image: {image_path}")

    # 2. Basic image + text similarity (CLIP-style, from README)
    texts = [
        "a frontal chest x-ray radiograph",
        "a lateral chest x-ray radiograph",
    ]
    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    # 3. Load MedCLIP-ViT model
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained()

    # Use GPU if available, else stay on CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for k in inputs:
        if isinstance(inputs[k], torch.Tensor):
            inputs[k] = inputs[k].to(device)

    print(f"Running model on device: {device}")

    outputs = model(**inputs)
    img_embeds = outputs["img_embeds"]        # [1, D]
    text_embeds = outputs["text_embeds"]      # [2, D]

    # 4. Cosine similarity between image and each text
    img_norm = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
    txt_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    sims = (img_norm @ txt_norm.T).squeeze(0)  # [2]

    print("Text prompts:", texts)
    print("Cosine similarities:", sims.tolist())
    pred_idx = int(torch.argmax(sims))
    print(f"Predicted view prompt: {texts[pred_idx]}")

    # 5. Optional: Prompt-based classification using CheXpert disease prompts
    print("\n--- PromptClassifier demo (CheXpert prompts) ---")
    cls_prompts = process_class_prompts(generate_chexpert_class_prompts(n=5))
    clf = PromptClassifier(model, ensemble=True).to(device)

    # Only image needed here
    img_only_inputs = processor(images=image, return_tensors="pt")
    for k in img_only_inputs:
        if isinstance(img_only_inputs[k], torch.Tensor):
            img_only_inputs[k] = img_only_inputs[k].to(device)
    img_only_inputs["prompt_inputs"] = cls_prompts

    out = clf(**img_only_inputs)
    print("CheXpert class names:", out["class_names"])
    print("CheXpert logits:", out["logits"].detach().cpu().tolist())

if __name__ == "__main__":
    main()