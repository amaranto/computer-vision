import io
import requests
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from transformers import DetrFeatureExtractor, DetrForSegmentation
from transformers.models.detr.feature_extraction_detr import rgb_to_id


url = "../media/auto.jpg"
image = Image.open(url)
font = ImageFont.truetype("arial.ttf", 25)

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)

processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
result = feature_extractor.post_process_panoptic(outputs, processed_sizes, threshold=0.85)[0]

panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
panoptic_seg_id = rgb_to_id(panoptic_seg)

palette = itertools.cycle(sns.color_palette())

segmented_image = Image.fromarray(np.zeros_like(panoptic_seg, dtype=np.uint8))
draw = ImageDraw.Draw(segmented_image)
segmented_image = segmented_image.resize((image.width, image.height))

COCO_LABELS = {
    1: 'persona', 2: 'bicicleta', 3: 'coche', 4: 'motocicleta',
    17: 'gato', 18: 'perro'
}

for segment_info in result["segments_info"]:
    class_id = segment_info["category_id"]
    class_name = COCO_LABELS.get(class_id, 'Desconocido') 
    id = segment_info["id"]
    
    mask = panoptic_seg_id == id
    color = np.array(next(palette)) * 255 
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    
    color_image = Image.new("RGB", segmented_image.size, color=tuple(color.astype(int)))
    
    if class_name == "coche" or class_name == "persona":
        mask_image = mask_image.resize((image.width, image.height))
        mask_image = mask_image.point(lambda x: 0 if x == 0 else 127)
        segmented_image.paste(color_image, (0,0), mask=mask_image)
        draw = ImageDraw.Draw(image)
        where = np.where(mask)
        if where[0].size > 0 and where[1].size > 0:
            x, y = np.min(where[1]), np.min(where[0])
            #draw.text((x, y),class_name, font=font, stroke_fill=True, stroke_width=10 ,fill=(255, 255, 255, 0), embedded_color=True)
        segmented_image = segmented_image.convert("RGBA")
        image.paste(color_image, (0, 0), mask_image)

image.save("../media_output/auto_segmented.jpg")
plt.figure(figsize=(30, 15))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title('Imagen Segmentada con Etiquetas')
plt.axis('off')

plt.show()