#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chinese-CLIP Baseline for Multimodal Retrieval

Usage:
  修改 data_dir / model_dir 路径，然后直接运行：
    python baseline.py
"""

import os, io, json, base64
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import ChineseCLIPModel, ChineseCLIPProcessor


# ===================== 基础工具 =====================
def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def read_images_tsv(tsv_path):
    """逐行读取 TSV (item_id, base64)"""
    ids, imgs = [], []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            item_id, b64 = parts[0], parts[-1]
            try:
                img = b64_to_pil(b64)
            except Exception:
                continue
            ids.append(item_id)
            imgs.append(img)
    return ids, imgs


def read_jsonl(path):
    return [json.loads(l) for l in open(path, "r", encoding="utf-8") if l.strip()]


def l2_normalize(x: np.ndarray, eps=1e-10):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


# ===================== 模型封装 =====================
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

#仅仅对于valid的数据而言：text_feats:[5008,512]。img_feats：【189585.512】，img_ids：【189585】。这里分别代表的是valid中关于query的向量；和所有图片的汇总的向量综合，以及对应的img_ids图像的ids。
# ===================== 检索与评测 =====================
def retrieve_topk(text_feats, img_feats, img_ids, topk=10):
    sims = text_feats @ img_feats.T
    idx = np.argpartition(-sims, kth=topk - 1, axis=1)[:, :topk]
    row = np.arange(sims.shape[0])[:, None]
    sorted_idx = np.argsort(-sims[row, idx], axis=1)
    top_idx = idx[row, sorted_idx]
    # 在这里强制转成 int
    return [[int(img_ids[j]) for j in row] for row in top_idx]


# def recall_at_10(valid_queries, qids, pred_ids):
#     gt = {q["query_id"]: set(q.get("item_ids", [])) for q in valid_queries}
#     hit, tot = 0, 0
#     for qid, pred in zip(qids, pred_ids):
#         if qid in gt and gt[qid]:#ground trueth and 取出来值不为kong
#             tot += 1
#             if any(pid in gt[qid] for pid in pred[:10]):
#                 hit += 1
#     return hit / max(tot, 1)
# 评估时强制转成 str
def recall_at_10(valid_queries, qids, pred_ids):
    gt = {q["query_id"]: set([int(x) for x in q.get("item_ids", [])]) for q in valid_queries}
    hit, tot = 0, 0
    for qid, pred in zip(qids, pred_ids):
        if qid in gt and gt[qid]:
            tot += 1
            if any(pid in gt[qid] for pid in pred[:10]):  # 这里 pred 已经是 int
                hit += 1
    return hit / max(tot, 1)
def write_submission(test_jsonl_in, qids, pred_ids, out_path):
    qid2pred = {q: p for q, p in zip(qids, pred_ids)}
    with open(test_jsonl_in, "r", encoding="utf-8") as f, \
         open(out_path, "w", encoding="utf-8") as w:
        for line in f:
            q = json.loads(line)
            qid = q.get("query_id")
            items = qid2pred.get(qid, [])[:10]
            items = [int(x) for x in items]  # 强制转 int
            items += [0] * (10 - len(items))  # 容错补齐
            q["item_ids"] = items
            w.write(json.dumps(q, ensure_ascii=False) + "\n")
import json
import numpy as np
import os

def build_or_load_img_split(tsv_path, split_tag, model, save_dir="embeddings", bs=64):
    """
    对应一个 MR_*_imgs.tsv：
    - 如果 embeddings/{split_tag}_img_feats.npy & ids.json 已存在 -> 直接加载
    - 否则：读取TSV、编码、保存缓存，再返回
    """
    os.makedirs(save_dir, exist_ok=True)
    feat_p = os.path.join(save_dir, f"{split_tag}_img_feats.npy")
    ids_p  = os.path.join(save_dir, f"{split_tag}_img_ids.json")

    if os.path.exists(feat_p) and os.path.exists(ids_p):
        feats = np.load(feat_p)
        with open(ids_p, "r", encoding="utf-8") as f:
            ids = json.load(f)
        print(f"[Cache] load {split_tag} images: {feats.shape[0]}")
        return ids, feats

    # 现算
    ids, imgs = read_images_tsv(tsv_path)
    feats = model.encode_images(imgs, bs=bs)
    np.save(feat_p, feats)
    with open(ids_p, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)
    print(f"[Save] {split_tag}: {len(ids)} -> {feat_p}")
    return ids, feats


def build_or_load_text_feats(jsonl_path, model, split_tag, save_dir="./embeddings", bs=128):
    """
    对应一个 MR_*_queries.jsonl：
    - 如果 embeddings/{split_tag}_txt_feats.npy & qids.json 已存在 -> 直接加载
    - 否则：读取jsonl、编码、保存缓存，再返回
    返回：texts(仅调试用)、qids、txt_feats
    """
    os.makedirs(save_dir, exist_ok=True)
    feat_p = os.path.join(save_dir, f"{split_tag}_txt_feats.npy")
    qids_p = os.path.join(save_dir, f"{split_tag}_qids.json")
    #query_text ; query_id
    queries = read_jsonl(jsonl_path)
    texts = [q.get("query") or q.get("query_text") for q in queries]
    qids  = [q.get("query_id") or q.get("query_id") for q in queries]

    if os.path.exists(feat_p) and os.path.exists(qids_p):
        feats = np.load(feat_p)
        with open(qids_p, "r", encoding="utf-8") as f:
            qids_loaded = json.load(f)
        # 防止数据变动导致错配
        if qids_loaded == qids and feats.shape[0] == len(qids):
            print(f"[Cache] load {split_tag} text feats: {feats.shape[0]}")
            return texts, qids, feats

    # 现算
    feats = model.encode_texts(texts, bs=bs)
    np.save(feat_p, feats)
    with open(qids_p, "w", encoding="utf-8") as f:
        json.dump(qids, f, ensure_ascii=False)
    print(f"[Save] {split_tag} text feats: {feats.shape[0]} -> {feat_p}")
    return texts, qids, feats

# ===================== 主流程 =====================
def main():
    # 需要你修改的路径
    data_dir = "/Volumes/PSSD/sources/02多模态图文检索赛/Multimodal_Retrieval"
    model_dir = "/Volumes/mac_win/models/AI-ModelScope/chinese-clip-vit-base-patch16"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPWrapper(model_dir, device=device)

    # 1) 合并 train+valid+test 图片库（带缓存）
    all_img_ids, all_img_feats = [], []
    for split_file in ["MR_train_imgs.tsv", "MR_valid_imgs.tsv", "MR_test_imgs.tsv"]:
        path = os.path.join(data_dir, split_file)
        if os.path.exists(path):
            tag = split_file.replace(".tsv", "")  # e.g. MR_train_imgs
            ids, feats = build_or_load_img_split(path, tag, model, save_dir="embeddings", bs=64)
            all_img_ids.extend(ids)
            all_img_feats.append(feats)

    all_img_feats = np.concatenate(all_img_feats, axis=0)
    print(f"[Index] total images: {len(all_img_ids)}  feats: {all_img_feats.shape}")
    # （可选）把合并后的全库再存一个总表，后面直接加载更快
    np.save("embeddings/all_img_feats.npy", all_img_feats)
    with open("embeddings/all_img_ids.json", "w", encoding="utf-8") as f:
        json.dump(all_img_ids, f, ensure_ascii=False)

    # 2) 验证集评测
    valid_q_path = os.path.join(data_dir, "MR_valid_queries.jsonl")
    if os.path.exists(valid_q_path):
        _, v_qids, v_txt_feats = build_or_load_text_feats(
            valid_q_path, model, split_tag="MR_valid_queries", save_dir="/Volumes/PSSD/NetThink/pythonProject/7-19-Project/Chinese-CLIP baseline/embeddings", bs=256
        )
        pred_ids = retrieve_topk(v_txt_feats, all_img_feats, all_img_ids, topk=10)
        valid_queries = read_jsonl(valid_q_path)
        r10 = recall_at_10(valid_queries, v_qids, pred_ids)
        print(f"[Valid] Recall@10 = {r10:.4f}")

    # 3) 测试集提交
    test_q_path = os.path.join(data_dir, "MR_test_queries.jsonl")
    if os.path.exists(test_q_path):
        _, t_qids, t_txt_feats = build_or_load_text_feats(
            test_q_path, model, split_tag="MR_test_queries", save_dir="/Volumes/PSSD/NetThink/pythonProject/7-19-Project/Chinese-CLIP baseline/embeddings", bs=256
        )
        pred_ids = retrieve_topk(t_txt_feats, all_img_feats, all_img_ids, topk=10)
        out_path = os.path.join(data_dir, "test_pred.jsonl")
        write_submission(test_q_path, t_qids, pred_ids, out_path)
        print(f"[Submit] 写出预测到 {out_path}")


if __name__ == "__main__":
    main()

'''
[Cache] load MR_train_imgs images: 129380
[Cache] load MR_valid_imgs images: 29806
[Cache] load MR_test_imgs images: 30399
[Index] total images: 189585  feats: (189585, 512)
[Cache] load MR_valid_queries text feats: 5008
[Valid] Recall@10 = 0.6607
[Cache] load MR_test_queries text feats: 5004

'''