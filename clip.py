import csv
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import roc_auc_score


processor = CLIPProcessor.from_pretrained('/data1/modes/transformers/clip-vit-base-patch32')
model = CLIPModel.from_pretrained('/data1/modes/transformers/clip-vit-base-patch32')
device = torch.device('cuda')
model.to(device)
model.eval()

labels, scores = [], []
with open('annotation.csv', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for i, row in enumerate(csv_reader):
        label = row['Label']
        text = row['Title']
        image = Image.open(f'images/{i}.jpg').convert('RGB')
        inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
        inputs.to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            text_embeds = outputs.text_embeds.squeeze()
            image_embeds = outputs.image_embeds.squeeze()
            score = 1.0 - F.cosine_similarity(text_embeds, image_embeds, dim=0).item()

        labels.append(label)
        scores.append(score)
        print(f'similarity score for sample #{i:03d}: {score:.2f}, label: {label}')

auc = roc_auc_score(labels, scores)
print(f'AUROC for CLIP: {auc:.2f}')
