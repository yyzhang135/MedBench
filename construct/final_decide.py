# -*- coding: utf-8 -*-
import json
import os
import hashlib
from tqdm import tqdm

BASE_DIR = " "
 
DECISION_PATH = os.path.join(BASE_DIR,"final_decision.jsonl")

NEG_REL_IN = os.path.join(BASE_DIR, "negative_relevance_set.jsonl")
NEG_REL_OUT = os.path.join(BASE_DIR,"negative_relevance_set_final.jsonl")
NEG_CORR_IN = os.path.join(BASE_DIR,"negative_doc_set.jsonl")
NEG_CORR_OUT = os.path.join(BASE_DIR,"negative_doc_set_final.jsonl")

POS_REL_IN = os.path.join(BASE_DIR,"positive_relevance_set.jsonl")
POS_REL_OUT = os.path.join(BASE_DIR,"positive_relevance_set_final.jsonl")
POS_CORR_IN = os.path.join(BASE_DIR,"positive_doc_set.jsonl")
POS_CORR_OUT = os.path.join(BASE_DIR,"positive_doc_set_final.jsonl")


def robust_hash(text):
    if not text: return ""
    return hashlib.md5(str(text).strip().encode('utf-8')).hexdigest()


def main():
    print("开始执行 MedBench 数据集物理更新...")

    to_remove_pair_hashes = set() 
    to_remove_doc_hashes = set()  
    to_add_items = []  

    if not os.path.exists(DECISION_PATH):
        print(f"错误: 找不到决策文件 {DECISION_PATH}")
        return

    with open(DECISION_PATH, 'r', encoding='utf-8',errors='ignore') as f:
        for line in f:
            try:
                item = json.loads(line)
                action = item.get("final_action")
                q = item.get("query", "").strip()
                d = item.get("document", "").strip()

                if action == "MOVE_TO_POSITIVE":
                    to_remove_pair_hashes.add(robust_hash(q + d))
                    to_remove_doc_hashes.add(robust_hash(d))
                    to_add_items.append({"q": q, "d": d})
                elif action == "ADD_NEW_POSITIVE":
                    to_add_items.append({"q": q, "d": d})
            except:
                continue

    print(f" 决策概览:")
    print(f"   需从负集中移除 (MOVE): {len(to_remove_pair_hashes)} 条")
    print(f"   需加入正集 (MOVE+ADD): {len(to_add_items)} 条")

    print("\n 正在清洗负样本...")

    if os.path.exists(NEG_REL_IN):
        neg_rel_removed = 0
        with open(NEG_REL_IN, 'r', encoding='utf-8',errors='ignore') as f_in, open(NEG_REL_OUT, 'w', encoding='utf-8') as f_out:
            for line in tqdm(f_in, desc="Cleaning Neg Rel"):
                item = json.loads(line)
                if robust_hash(item.get("query", "") + item.get("doc", "")) in to_remove_pair_hashes:
                    neg_rel_removed += 1
                    continue
                f_out.write(line)
        print(f"  负向标签集清洗完成，移除 {neg_rel_removed} 条。")

        if os.path.exists(NEG_CORR_IN):
            neg_doc_removed = 0
            with open(NEG_CORR_IN, 'r', encoding='utf-8', errors='ignore') as f_in, open(NEG_CORR_OUT, 'w',encoding='utf-8') as f_out:
                for line in tqdm(f_in, desc="Cleaning Neg Docs"):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if robust_hash(item.get("doc", "")) in to_remove_doc_hashes:
                        neg_doc_removed += 1
                        continue
                    f_out.write(line + "\n")  
            print(f"  负向文档集清洗完成，移除 {neg_doc_removed} 条。")

    print("\n 正在增强正样本...")

    if os.path.exists(POS_REL_IN):
        with open(POS_REL_IN, 'r', encoding='utf-8',errors='ignore') as f_in, open(POS_REL_OUT, 'w', encoding='utf-8') as f_out:
            for line in f_in: f_out.write(line) 
            for item in tqdm(to_add_items, desc="Appending Pos Rel"):
                new_entry = {
                    "query": item["q"],
                    "doc": item["d"],
                    "label": 1,
                    "scene_label": "Auto_Corrected"
                }
                f_out.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
        print(f"   正向标签集更新完成。")

    if os.path.exists(POS_CORR_IN):
        with open(POS_CORR_IN, 'r', encoding='utf-8',errors='ignore') as f_in, open(POS_CORR_OUT, 'w', encoding='utf-8') as f_out:
            for line in f_in: f_out.write(line) 
            for item in tqdm(to_add_items, desc="Appending Pos Docs"):
                new_entry = {"doc": item["d"]}
                f_out.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
        print(f"  正向文档集更新完成。")

    print("\n" + "=" * 50)
    print("任务完成！")
    print(f" 新负样本集: {os.path.basename(NEG_REL_OUT)}")
    print(f" 新正样本集: {os.path.basename(POS_REL_OUT)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
