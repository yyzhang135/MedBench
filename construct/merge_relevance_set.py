# -*- coding: utf-8 -*-
import os
import json
from tqdm import tqdm

BASE_DIR = " "
TEST_DATASET_DIR = os.path.join(BASE_DIR)

NEG_FILE = os.path.join(TEST_DATASET_DIR, "negative_relevance_set_final.jsonl")
POS_FILE = os.path.join(TEST_DATASET_DIR, "positive_relevance_set_final.jsonl")

OUTPUT_FILE = os.path.join(TEST_DATASET_DIR, "test_qrels.jsonl")

def merge_qrels():
    print("开始合并正负样本，生成最终 Qrels 测试集...")

    if not os.path.exists(NEG_FILE):
        print(f" 错误: 找不到负样本文件 {NEG_FILE}")
        return
    if not os.path.exists(POS_FILE):
        print(f" 错误: 找不到正样本文件 {POS_FILE}")
        return

    total_merged = 0
    neg_count = 0
    pos_count = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:

        print("\n 正在合并负样本...")
        with open(NEG_FILE, 'r', encoding='utf-8', errors='ignore') as f_neg:
            for line in tqdm(f_neg, desc="Merging Negatives"):
                line = line.strip()
                if not line: continue 

                try:
                    json.loads(line)
                    f_out.write(line + "\n")
                    neg_count += 1
                    total_merged += 1
                except json.JSONDecodeError:
                    continue  

        print("\n 正在合并正样本...")
        with open(POS_FILE, 'r', encoding='utf-8', errors='ignore') as f_pos:
            for line in tqdm(f_pos, desc="Merging Positives"):
                line = line.strip()
                if not line: continue

                try:
                    json.loads(line)
                    f_out.write(line + "\n")
                    pos_count += 1
                    total_merged += 1
                except json.JSONDecodeError:
                    continue

    print("\n" + "=" * 50)
    print(f" Qrels 测试集生成完毕！")
    print(f"数据统计:")
    print(f"   负样本数量: {neg_count} 条")
    print(f"   正样本数量: {pos_count} 条")
    print(f"   总计评测对: {total_merged} 条")
    print(f"文件已保存至: {OUTPUT_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    merge_qrels()
