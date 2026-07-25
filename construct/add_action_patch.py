import json
import os

BASE_DIR = " "

VOTED_FILE = os.path.join(BASE_DIR, "voted_pre_labeled_candidates.jsonl")

NEG_FILE = os.path.join(BASE_DIR, "negative_relevance_set.jsonl")

OUTPUT_FILE = os.path.join(BASE_DIR,"voted_pre_labeled_candidates_with_actions.jsonl")

def main():
    print(f"正在加载负样本库: {NEG_FILE} ...")
    hard_neg_map = {}
    if os.path.exists(NEG_FILE):
        with open(NEG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item.get("label") == 0:
                        q = item.get("query", "").strip()
                        d = item.get("doc", "").strip()
                        if q and d:
                            if q not in hard_neg_map: hard_neg_map[q] = set()
                            hard_neg_map[q].add(hash(d))
                except:
                    continue
        print(f"成功加载了 {len(hard_neg_map)} 个查询的负样本比对集。")
    else:
        print("找不到负样本文件！")
        return

    print(f"正在给投票结果打补丁: {VOTED_FILE} ...")
    processed_count = 0
    hard_negative_found = 0  

    with open(VOTED_FILE, 'r', encoding='utf-8') as fin, \
            open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:

        for line in fin:
            data = json.loads(line.strip())
            query = data.get("query", "")
            current_hard_negs = hard_neg_map.get(query, set())

            for cand in data.get("candidates", []):
                doc = cand.get("document", "")
                votes = cand.get("votes", 0)
                is_hard_neg = hash(doc) in current_hard_negs

                if is_hard_neg:
                    cand["type"] = "Hard_Negative"
                    cand["suggested_action"] = "CHECK_TO_RECLASSIFY_AS_POSITIVE"
                    cand["reason"] = f"Conflict detected: Hard Negative ranked high by {votes}/3 models. Suggest REMOVE from negative set AND ADD to positive set."
                    hard_negative_found += 1
                else:
                    cand["type"] = "Unknown"
                    cand["suggested_action"] = "CHECK_TO_ADD"
                    cand["reason"] = f"Unknown Doc ranked high by {votes}/3 models"

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            processed_count += 1

    print(f"结果已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
