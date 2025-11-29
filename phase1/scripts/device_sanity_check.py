import torch
from medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT

# 1) CPU test
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()
model.to("cpu").eval()

dummy_pixels = torch.randn(1, 3, 224, 224)
dummy_ids = torch.randint(0, 1000, (2, 16))
dummy_mask = torch.ones_like(dummy_ids)

out = model(input_ids=dummy_ids, attention_mask=dummy_mask, pixel_values=dummy_pixels)
print(out["img_embeds"].shape, out["text_embeds"].shape)

# 2) (On Kaggle) GPU test
if torch.cuda.is_available():
    model.to("cuda").eval()
    out_gpu = model(
        input_ids=dummy_ids.to("cuda"),
        attention_mask=dummy_mask.to("cuda"),
        pixel_values=dummy_pixels.to("cuda"),
    )
    print(out_gpu["img_embeds"].shape, out_gpu["text_embeds"].shape)
