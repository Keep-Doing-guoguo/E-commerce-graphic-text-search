#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/9/22 10:59
@source from: 
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import io
import json
import base64
from typing import Dict, Iterator, Tuple, List
from PIL import Image

DATA_DIR = "/Volumes/PSSD/sources/02多模态图文检索赛/Multimodal_Retrieval"   # <<< 修改为你的解压目录
SAVE_SAMPLE_IMG = True                       # 是否把样例图片保存出来
SAMPLE_IMG_DIR = "./_preview_imgs"           # 样例图片输出目录
NUM_SHOW = 3                                 # 每类展示多少条

def iter_img_tsv(tsv_path: str) -> Iterator[Tuple[str, str]]:
    """逐行读取 TSV，yield (item_id, b64str)"""
    with open(tsv_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                # 某些数据可能含有多列，这里取第一列为id、最后一列为b64
                # 但如果确实不符合格式，跳过这行
                print(f"[WARN] bad line #{ln} in {os.path.basename(tsv_path)}: {line[:80]}...")
                continue
            item_id = parts[0]
            b64 = parts[-1]
            yield item_id, b64

def decode_b64_to_pil(b64str: str) -> Image.Image:
    """base64 -> PIL.Image"""
    return Image.open(io.BytesIO(base64.b64decode(b64str))).convert("RGB")

def read_jsonl(jsonl_path: str) -> List[Dict]:
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] JSONL parse error line #{ln}: {e}")
    return data

def preview_imgs(split_name: str, fname: str):
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"[INFO] {fname} not found, skip.")
        return
    print(f"\n=== Preview {split_name} images from {fname} ===")
    count = 0
    os.makedirs(SAMPLE_IMG_DIR, exist_ok=True)
    for i, (item_id, b64) in enumerate(iter_img_tsv(path)):
        if i < NUM_SHOW:
            try:
                img = decode_b64_to_pil(b64)
                print(f"[{split_name} img {i}] item_id={item_id}, size={img.size}, mode={img.mode}")
                if SAVE_SAMPLE_IMG:
                    out = os.path.join(SAMPLE_IMG_DIR, f"{split_name}_{i}_{item_id}.jpg")
                    img.save(out)
            except Exception as e:
                print(f"[WARN] decode error item_id={item_id}: {e}")
        count += 1
    print(f"[{split_name}] total image rows: {count}")

def preview_queries(split_name: str, fname: str):
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"[INFO] {fname} not found, skip.")
        return
    print(f"\n=== Preview {split_name} queries from {fname} ===")
    data = read_jsonl(path)
    print(f"[{split_name}] total queries: {len(data)}")
    for i, q in enumerate(data[:NUM_SHOW]):
        # 常见字段：query_id / query / item_ids(在train/valid可能存在，test通常为空)
        keys = list(q.keys())
        print(f"[{split_name} query {i}] keys={keys}")
        print(f"  -> query_id: {q.get('query_id', q.get('qid'))}")
        print(f"  -> query:    {q.get('query', q.get('text'))}")
        if "item_ids" in q:
            print(f"  -> gt item_ids (len={len(q['item_ids'])}): {q['item_ids'][:5]}{'...' if len(q['item_ids'])>5 else ''}")

def main():
    # 预览 train/valid/test 的图片
    preview_imgs("train", "MR_train_imgs.tsv")
    preview_imgs("valid", "MR_valid_imgs.tsv")
    preview_imgs("test",  "MR_test_imgs.tsv")

    # 预览 train/valid/test 的 queries
    preview_queries("train", "MR_train_queries.jsonl")
    preview_queries("valid", "MR_valid_queries.jsonl")
    preview_queries("test",  "MR_test_queries.jsonl")

    print("\nDone. 样例图片（若开启）已输出到：", os.path.abspath(SAMPLE_IMG_DIR))

if __name__ == "__main__":
    main()