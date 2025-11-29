import torch
from PIL import Image

from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor

# 🔁 TODO: change this to any IU-CXR image you have
IMG_PATH = r"D:\MedVLMBench\phase1\data\chestxray_iu\images\images_normalized\3806_IM-1916-2001.dcm.png"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print("Loading MedCLIP model...")
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained()
    model.to(device).eval()

    processor = MedCLIPProcessor()

    # 1) Image → embedding
    image = Image.open(IMG_PATH).convert("RGB")
    img_inputs = processor(images=image, return_tensors="pt")
    pixel_values = img_inputs["pixel_values"]          # [1, 3, H, W]

    with torch.no_grad():
        img_embeds = model.vision_model(pixel_values, project=True)   # [1, 512]
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

    # 2) Text → embedding
    prompts = [
        "a frontal chest x-ray radiograph",
        "a lateral chest x-ray radiograph",
    ]
    txt_inputs = processor(text=prompts, return_tensors="pt", padding=True)
    input_ids = txt_inputs["input_ids"]               # [2, L]
    attention_mask = txt_inputs["attention_mask"]     # [2, L]

    with torch.no_grad():
        text_embeds = model.text_model(input_ids=input_ids,
                                       attention_mask=attention_mask)  # [2, 512]
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # 3) Similarity
    logits = img_embeds @ text_embeds.T              # [1, 2]
    pred_idx = logits.argmax(dim=-1).item()
    classes = ["Frontal", "Lateral"]

    print("Logits:", logits.numpy())
    print("Predicted view:", classes[pred_idx])

if __name__ == "__main__":
    main()