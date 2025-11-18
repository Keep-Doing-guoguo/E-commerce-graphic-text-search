#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/9/22 11:31
@source from: 
"""
import torch
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor  # ✅ 注意这里

# 可以用本地路径或在线仓库名
model_dir = "/Volumes/mac_win/models/AI-ModelScope/chinese-clip-vit-base-patch16"  # 或 "OFA-Sys/chinese-clip-vit-base-patch16"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChineseCLIPModel.from_pretrained(model_dir).to(device).eval()
processor = ChineseCLIPProcessor.from_pretrained(model_dir)

image = Image.open("/Volumes/PSSD/NetThink/pythonProject/GITHUB/E-commerce-graphic-text-search/椅子.png")  # 换成你的图片
texts = ["人", "有人在椅子坐着", "一个人在抽烟"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)  # [1, num_texts]

for t, p in zip(texts, probs[0].tolist()):
    print(f"{t} -> {p:.4f}")