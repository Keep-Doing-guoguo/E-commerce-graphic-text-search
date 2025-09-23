#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/9/22 10:18
@source from: 
"""
import pandas as pd, base64, io
from PIL import Image

def read_imgs_tsv(tsv_path):
    # 常见两列：item_id \t base64
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["item_id","b64"])
    return df

def b64_to_pil(b64s):
    img_bytes = base64.b64decode(b64s)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

import json

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
path = "/Volumes/PSSD/sources/02多模态图文检索赛/Multimodal_Retrieval/MR_train_imgs.tsv"
print(read_jsonl(path))
# 训练/验证：通常有 "query", "item_ids"(正例)
# 测试：      有 "query"，需要你填充 "item_ids"(长度=10)