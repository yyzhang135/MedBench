# -*- coding: utf-8 -*-
import json
import os
from tqdm import tqdm

BASE_DIR = " "

INPUT_PATH = os.path.join(BASE_DIR, "voted_pre_labeled_candidates_with_actions.jsonl")

OUTPUT_PATH = os.path.join(BASE_DIR, "data", "flattened_candidates.jsonl")

def main():
    print(f"源文件: {INPUT_PATH}")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"找不到输入文件: {INPUT_PATH}")

    total_pairs = 0

    with open(INPUT_PATH, 'r', encoding='utf-8') as f_in, \
            open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:

        for line_idx, line in enumerate(tqdm(f_in, desc="Flattening")):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            query = item.get("query", "")
            candidates = item.get("candidates", [])

            for cand_idx, cand in enumerate(candidates):
                flat_record = {
                    "group_id": line_idx,      
                    "inner_idx": cand_idx,     
                    "query": query,
                    "document": cand.get("document", ""),
                    "type": cand.get("type", "Unknown"), 
                    "suggested_action": cand.get("suggested_action", ""),  
                    "votes": cand.get("votes", 0),                        
                    "reason": cand.get("reason", ""),                     
                    "avg_rerank_score": cand.get("avg_rerank_score", 0.0)   
                }

                f_out.write(json.dumps(flat_record, ensure_ascii=False) + "\n")
                total_pairs += 1

    print(f"\n处理完成！")
    print(f"结果保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
