#!/usr/bin/env python3
# coding: utf-8
"""
simple_clip_search.py

Minimal demo using the provided CLIPWrapper:

- load images from a folder (by filename)
- encode images and a list of text queries
- compute cosine (dot after L2 norm) similarities and return top-k per query

Usage:
    修改 IMAGE_DIR / MODEL_DIR / QUERIES，然后运行：
        python simple_clip_search.py
"""
import json
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

# ------------------ your CLIPWrapper (kept exactly as you provided) ------------------
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

class CLIPWrapper:
    def __init__(self, model_dir, device="cuda"):
        self.device = device
        self.model = ChineseCLIPModel.from_pretrained(model_dir).to(device).eval()
        self.processor = ChineseCLIPProcessor.from_pretrained(model_dir)

    @torch.no_grad()
    def encode_images(self, pil_list, bs=64):
        feats = []
        for i in range(0, len(pil_list), bs):
            batch = pil_list[i:i + bs]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(self.device)
            f = self.model.get_image_features(**inputs)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
        return torch.cat(feats, 0).numpy()

    @torch.no_grad()
    def encode_texts(self, texts, bs=128):
        feats = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            f = self.model.get_text_features(**inputs)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
        return torch.cat(feats, 0).numpy()
# -------------------------------------------------------------------------

# ------------------ helper utilities ------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(folder, max_files=None):
    files = []
    for fn in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, fn)):
            continue
        ext = os.path.splitext(fn)[1].lower()
        if ext in IMG_EXTS:
            files.append(fn)
    files.sort()
    if max_files:
        files = files[:max_files]
    return files

def load_images(folder, filenames):
    pil_list = []
    for fn in tqdm(filenames, desc="Loading images"):
        path = os.path.join(folder, fn)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"skip {fn}: {e}")
            continue
        pil_list.append(img)
    return pil_list

def retrieve_topk(text_feats, img_feats, topk=10):
    # both should be numpy arrays, already L2-normalized
    # compute sims and get topk indices
    sims = np.matmul(text_feats, img_feats.T)  # (Q, N)
    idx = np.argpartition(-sims, kth=topk-1, axis=1)[:, :topk]
    row = np.arange(sims.shape[0])[:, None]
    sorted_idx = np.argsort(-sims[row, idx], axis=1)
    top_idx = idx[row, sorted_idx]  # shape (Q, topk)
    return top_idx  # indices into image list
# -------------------------------------------------------

def main():
    # ====== modify these ======
    IMAGE_DIR = "/path/to/your/image_folder"     # 你的图片目录
    MODEL_DIR = "/path/to/chinese-clip-model"    # Chinese-CLIP 模型目录
    MAX_IMAGES = None    # None or int (如果只想测试前 N 张)
    TOPK = 5
    # example queries (改成你的 100 个 query 列表)
    QUERIES = [
        "纯棉碎花吊带裙",
        "北欧轻奢边几",
        "红色连衣裙",
        # ... 其余 query
    ]
    # ==========================

    assert os.path.isdir(IMAGE_DIR), f"IMAGE_DIR not found: {IMAGE_DIR}"
    assert os.path.isdir(MODEL_DIR), f"MODEL_DIR not found: {MODEL_DIR}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    model = CLIPWrapper(MODEL_DIR, device=device)

    # 1) list & load images
    filenames = list_images(IMAGE_DIR, max_files=MAX_IMAGES)
    if len(filenames) == 0:
        print("no images found in", IMAGE_DIR)
        return

    pil_imgs = load_images(IMAGE_DIR, filenames)
    print(f"loaded {len(pil_imgs)} images")

    # 2) encode images
    print("encoding images ...")
    img_feats = model.encode_images(pil_imgs, bs=32)  # (N, d), L2-normalized
    print("img_feats shape:", img_feats.shape)

    # 3) encode texts
    print("encoding texts ...")
    txt_feats = model.encode_texts(QUERIES, bs=64)    # (Q, d), L2-normalized
    print("txt_feats shape:", txt_feats.shape)

    # 4) retrieve topk
    top_idx = retrieve_topk(txt_feats, img_feats, topk=TOPK)
    for qi, q in enumerate(QUERIES):
        inds = top_idx[qi]
        ids = [filenames[i] for i in inds]
        print(f"\nQuery [{qi}] {q} -> top-{TOPK}:")
        for rank, fn in enumerate(ids, start=1):
            print(f"  {rank}. {fn}")

    # (optional) 写出结果
    outp = os.path.join(IMAGE_DIR, "pred_topk.jsonl")
    with open(outp, "w", encoding="utf-8") as w:
        for qi, q in enumerate(QUERIES):
            ids = [filenames[int(i)] for i in top_idx[qi]]
            w.write(json.dumps({"query": q, "topk": ids}, ensure_ascii=False) + "\n")
    print("wrote results to", outp)

if __name__ == "__main__":
    main()