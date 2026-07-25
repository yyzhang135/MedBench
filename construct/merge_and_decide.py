# -*- coding: utf-8 -*-
import json
import os
from tqdm import tqdm

BASE_DIR = " "

FLAT_PATH = os.path.join(BASE_DIR,"flattened_candidates.jsonl")
CLEAN_LLM_PATH = os.path.join(BASE_DIR,"LLM_output.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "final_decision.jsonl")

def main():
    print("正在加载候选集元数据...")
    meta_map = {}
    with open(FLAT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                idx = int(str(item.get("inner_idx", item.get("index"))).strip())
                key = (item["query"].strip(), idx)
                meta_map[key] = item
            except:
                continue

    print(f" 元数据加载完成: {len(meta_map)} 条")

    print(" 正在执行 MedBench 标签终审...")
    stats = {"MOVE_TO_POSITIVE": 0, "ADD_NEW_POSITIVE": 0}

    with open(CLEAN_LLM_PATH, 'r', encoding='utf-8') as f_in, \
            open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc="Processing LLM Labels"):
            try:
                res = json.loads(line)
                key = (res["query"].strip(), int(res["inner_idx"]))

                meta = meta_map.get(key)
                if not meta:
                    continue

                is_relevant = (int(res["label"]) == 1)
                if not is_relevant:
                    continue

                action = None
                reason = ""
                o_type = meta.get("type", meta.get("type"))

                if o_type == "Hard_Negative":
                    action = "MOVE_TO_POSITIVE"
                    reason = "Label Correction: Reclassified from Hard Negative to Positive by LLM consensus."
                elif o_type == "Unknown":
                    action = "ADD_NEW_POSITIVE"
                    reason = "Data Discovery: Newly identified positive sample by LLM consensus."

                if action:
                    stats[action] += 1
                    out_record = {
                        "query": meta["query"],
                        "document": meta["document"],
                        "final_action": action,    
                        "type": o_type,   
                        "reason": reason,         
                        "metadata": {
                            "group_id": meta.get("group_id"),
                            "inner_idx": key[1]
                        }
                    }
                    f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            except Exception as e:
                continue

    print("\n" + "="*40)
    print("📊 MedBench 标签修正统计结果:")
    print(f"🔹 需搬移的假负样本 (MOVE): {stats['MOVE_TO_POSITIVE']} 条")
    print(f"🔹 需新增的正样本 (ADD):  {stats['ADD_NEW_POSITIVE']} 条")
    print(f"📂 最终决策列表已保存至: {OUTPUT_PATH}")
    print("="*40)

if __name__ == "__main__":
    main()
