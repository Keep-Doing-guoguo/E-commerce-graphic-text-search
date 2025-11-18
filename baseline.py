#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/11/2 21:13
@source from:
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chinese-CLIP Baseline for Multimodal Retrieval (per-split image index)

说明：
- train 的 queries 只在 MR_train_imgs.tsv 对应的图像库中检索
- valid 的 queries 只在 MR_valid_imgs.tsv 对应的图像库中检索
- test 的 queries 只在 MR_test_imgs.tsv 对应的图像库中检索
- 尝试按 "base + adapter" 加载模型（如果 ADAPTER_DIR 存在且 peft 可用）
- 路径已“写死”在 main() 中，直接运行即可
"""

import os, io, json, base64
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

# try to import PeftModel for base+adapter inference
try:
    from peft import PeftModel
    _has_peft = True
except Exception:
    PeftModel = None
    _has_peft = False


# ===================== 基础工具 =====================
def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def read_images_tsv(tsv_path):
    """逐行读取 TSV (item_id, base64) -> 返回 ids(list[str]), imgs(list[PIL.Image])"""
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


# ===================== 模型封装（base + optional adapter） =====================
class CLIPWrapper:
    def __init__(self, base_model_dir, adapter_dir=None, device="cuda"):
        self.device = device
        # load base
        print(f"[Model] loading base from {base_model_dir} ...")
        base = ChineseCLIPModel.from_pretrained(base_model_dir)
        # if adapter exists and peft available, load adapter on top of base
        if adapter_dir and _has_peft and os.path.exists(adapter_dir):
            try:
                print(f"[Model] loading LoRA adapter from {adapter_dir} (PeftModel)...")
                peft_model = PeftModel.from_pretrained(base, adapter_dir)
                self.model = peft_model.to(device).eval()
                print("[Model] loaded base + adapter (PeftModel).")
            except Exception as e:
                print(f"[Model] failed to load adapter via PeftModel: {e}. Fallback to base only.")
                self.model = base.to(device).eval()
        else:
            if adapter_dir and not _has_peft:
                print("[Model] peft not installed; cannot load adapter. Using base only.")
            elif adapter_dir and not os.path.exists(adapter_dir):
                print(f"[Model] adapter dir not found ({adapter_dir}); using base only.")
            self.model = base.to(device).eval()

        self.processor = ChineseCLIPProcessor.from_pretrained(base_model_dir)

    @torch.no_grad()
    def encode_images(self, pil_list, bs=64):
        feats = []
        for i in range(0, len(pil_list), bs):
            batch = pil_list[i:i + bs]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            # keep tensors on CPU until moved to device in model call (.to will be done inside .get_image_features if model on device)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            f = self.model.get_image_features(**inputs)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
        return torch.cat(feats, 0).numpy() if feats else np.zeros((0, self.model.config.projection_dim), dtype=np.float32)

    @torch.no_grad()
    def encode_texts(self, texts, bs=128):
        feats = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            f = self.model.get_text_features(**inputs)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
        return torch.cat(feats, 0).numpy() if feats else np.zeros((0, self.model.config.projection_dim), dtype=np.float32)


# ===================== 检索与评测 =====================
def retrieve_topk(text_feats, img_feats, img_ids, topk=10):
    """
    text_feats: [N, D], img_feats: [M, D], img_ids: list length M (string ids)
    返回：每条 query 的 topk img_id (as int)
    """
    sims = text_feats @ img_feats.T   # [N, M]
    idx = np.argpartition(-sims, kth=topk - 1, axis=1)[:, :topk]   # [N, topk]
    row = np.arange(sims.shape[0])[:, None]
    sorted_idx = np.argsort(-sims[row, idx], axis=1)
    top_idx = idx[row, sorted_idx]
    # 安全转换为 int（当 id 不是纯数字时会触发 ValueError -> fallback keep original string）
    out = []
    for r in top_idx:
        row_ids = []
        for j in r:
            sid = img_ids[j]
            try:
                row_ids.append(int(sid))
            except Exception:
                row_ids.append(sid)
        out.append(row_ids)
    return out


def recall_at_10(valid_queries, qids, pred_ids):
    """
    valid_queries: list of dict (contains item_ids)
    qids: list of qid aligned with pred_ids
    pred_ids: list of list (items as int or str)
    将 GT 转为 int set（如果可转），pred 已经为 int(优先)或 str。
    """
    # build gt map: qid -> set(int or str)
    gt = {}
    for q in valid_queries:
        qid = q.get("query_id")
        raw = q.get("item_ids", [])
        converted = set()
        for x in raw:
            try:
                converted.add(int(x))
            except Exception:
                converted.add(x)
        gt[qid] = converted

    hit, tot = 0, 0
    for qid, preds in zip(qids, pred_ids):
        if qid in gt and gt[qid]:
            tot += 1
            # if any predicted id (int or str) in GT set
            found = False
            for p in preds[:10]:
                if p in gt[qid]:
                    found = True
                    break
                # also try string/int cross-compare
                try:
                    if str(p) in set(map(str, gt[qid])):
                        found = True
                        break
                except Exception:
                    pass
            if found:
                hit += 1
    return hit / max(tot, 1)


def write_submission(test_jsonl_in, qids, pred_ids, out_path):
    """
    写出 test 预测，保持 item_ids 为 int where possible；不够长度补 0
    """
    qid2pred = {q: p for q, p in zip(qids, pred_ids)}
    with open(test_jsonl_in, "r", encoding="utf-8") as f, \
         open(out_path, "w", encoding="utf-8") as w:
        for line in f:
            q = json.loads(line)
            qid = q.get("query_id")
            items = qid2pred.get(qid, [])[:10]
            # normalize: try int
            items_norm = []
            for it in items:
                try:
                    items_norm.append(int(it))
                except Exception:
                    items_norm.append(it)
            # pad
            if len(items_norm) < 10:
                pad_val = 0
                items_norm += [pad_val] * (10 - len(items_norm))
            q["item_ids"] = items_norm
            w.write(json.dumps(q, ensure_ascii=False) + "\n")


# ===================== cache helpers =====================
def build_or_load_img_split(tsv_path, split_tag, model, save_dir="embeddings", bs=64):
    """
    针对单个 split 的 images.tsv，计算并缓存它的图片向量与 ids
    返回 (ids_list, feats_np)
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


def build_or_load_text_feats(jsonl_path, model, split_tag, save_dir="embeddings", bs=128):
    """
    计算某 split 的文本特征并缓存（或从缓存加载）
    返回 (texts, qids, feats_np)
    """
    os.makedirs(save_dir, exist_ok=True)
    feat_p = os.path.join(save_dir, f"{split_tag}_txt_feats.npy")
    qids_p = os.path.join(save_dir, f"{split_tag}_qids.json")

    queries = read_jsonl(jsonl_path)
    texts = [q.get("query") or q.get("query_text") for q in queries]
    qids  = [q.get("query_id") for q in queries]

    if os.path.exists(feat_p) and os.path.exists(qids_p):
        feats = np.load(feat_p)
        with open(qids_p, "r", encoding="utf-8") as f:
            qids_loaded = json.load(f)
        if qids_loaded == qids and feats.shape[0] == len(qids):
            print(f"[Cache] load {split_tag} text feats: {feats.shape[0]}")
            return texts, qids, feats

    feats = model.encode_texts(texts, bs=bs)
    np.save(feat_p, feats)
    with open(qids_p, "w", encoding="utf-8") as f:
        json.dump(qids, f, ensure_ascii=False)
    print(f"[Save] {split_tag} text feats: {feats.shape[0]} -> {feat_p}")
    return texts, qids, feats


# ===================== 主流程 =====================
def main():
    # 写死路径（按你之前使用的路径）
    data_dir = "/home/model/public/real_zhangguowen/other_code/22_chinese_clip/Multimodal_Retrieval"
    base_model_dir = "/home/model/public/real_zhangguowen/models/AI-ModelScope/chinese-clip-vit-huge-patch14"
    # 如果你有 LoRA adapter（PEFT 输出的目录），写在这里；不存在就设为 None
    ADAPTER_DIR = ""  # <- 可改为实际 adapter 路径或 None /models/other_code/22_chinese_clip/checkpoints/clip_lora_single_gpu/epoch-5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Main] device={device}, base_model={base_model_dir}, adapter={ADAPTER_DIR}")
    model = CLIPWrapper(base_model_dir, adapter_dir=ADAPTER_DIR if os.path.exists(ADAPTER_DIR) else None, device=device)

    # === train split (只在 train images 中检索/计算) ===
    train_imgs_p = os.path.join(data_dir, "MR_train_imgs.tsv")
    train_q_p = os.path.join(data_dir, "MR_train_queries.jsonl")
    if os.path.exists(train_imgs_p) and os.path.exists(train_q_p):
        print("[Main] Preparing train split features ...")
        train_img_ids, train_img_feats = build_or_load_img_split(train_imgs_p, "MR_train_imgs", model, save_dir="embeddings", bs=64)
        # 若需要对 train queries 做检索/调试，可以像下面一样计算文本特征并检索（这里不强制执行）
        # train_texts, train_qids, train_txt_feats = build_or_load_text_feats(train_q_p, model, "MR_train_queries", save_dir="embeddings", bs=256)
        # train_pred = retrieve_topk(train_txt_feats, train_img_feats, train_img_ids, topk=10)
        # print("[Main] train done (example)")

    # === valid split（仅在 valid images 中检索） ===
    valid_imgs_p = os.path.join(data_dir, "MR_valid_imgs.tsv")
    valid_q_p = os.path.join(data_dir, "MR_valid_queries.jsonl")
    if os.path.exists(valid_imgs_p) and os.path.exists(valid_q_p):
        print("[Main] Preparing valid split features ...")
        valid_img_ids, valid_img_feats = build_or_load_img_split(valid_imgs_p, "MR_valid_imgs", model, save_dir="embeddings", bs=64)
        v_texts, v_qids, v_txt_feats = build_or_load_text_feats(valid_q_p, model, "MR_valid_queries", save_dir="embeddings", bs=256)
        pred_ids = retrieve_topk(v_txt_feats, valid_img_feats, valid_img_ids, topk=10)
        valid_queries = read_jsonl(valid_q_p)
        r10 = recall_at_10(valid_queries, v_qids, pred_ids)
        print(f"[Valid] Recall@10 = {r10:.4f}")

    # === test split（仅在 test images 中检索） ===
    test_imgs_p = os.path.join(data_dir, "MR_test_imgs.tsv")
    test_q_p = os.path.join(data_dir, "MR_test_queries.jsonl")
    if os.path.exists(test_imgs_p) and os.path.exists(test_q_p):
        print("[Main] Preparing test split features ...")
        test_img_ids, test_img_feats = build_or_load_img_split(test_imgs_p, "MR_test_imgs", model, save_dir="embeddings", bs=64)
        t_texts, t_qids, t_txt_feats = build_or_load_text_feats(test_q_p, model, "MR_test_queries", save_dir="embeddings", bs=256)
        pred_ids = retrieve_topk(t_txt_feats, test_img_feats, test_img_ids, topk=10)
        out_path = os.path.join(data_dir, "test_pred-large.jsonl")
        write_submission(test_q_p, t_qids, pred_ids, out_path)
        print(f"[Submit] 写出预测到 {out_path}")


if __name__ == "__main__":
    main()