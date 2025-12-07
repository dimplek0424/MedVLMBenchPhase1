def predict_diseases_for_image(img_path, processor, model, text_embeds):
    """
    Returns logits for 14 diseases for a single image.
    """
    img = Image.open(img_path).convert("RGB")
    img_inputs = processor(images=img, return_tensors="pt")
    pixel_values = img_inputs["pixel_values"]

    with torch.no_grad():
        img_embeds = model.vision_model(pixel_values, project=True)
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

        logits = img_embeds @ text_embeds.T   # shape: [1, num_classes]

    return logits.squeeze(0).cpu().numpy()
