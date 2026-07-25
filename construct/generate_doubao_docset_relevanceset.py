import json
import os
from tqdm import tqdm


input_file = 'LLM_output.jsonl'

output_dir = 'dataset'
neg_doc_file = os.path.join(output_dir, 'negative_doc_set.jsonl')
pos_doc_file = os.path.join(output_dir, 'positive_doc_set.jsonl')
neg_rel_file = os.path.join(output_dir, 'negative_relevance_set.jsonl')
pos_rel_file = os.path.join(output_dir, 'positive_relevance_set.jsonl')


def generate_datasets():
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        return

    os.makedirs(output_dir, exist_ok=True)

    negative_docs_set = set()
    positive_docs_set = set()

    neg_rel_count = 0
    pos_rel_count = 0

    print(f"开始解析 {input_file} 并生成相关性数据集...")

    with open(input_file, 'r', encoding='utf-8') as fin, \
            open(neg_rel_file, 'w', encoding='utf-8') as f_neg_rel, \
            open(pos_rel_file, 'w', encoding='utf-8') as f_pos_rel:

        for line in tqdm(fin, desc="处理进度"):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                scene_label_full = data.get("场景标签", "").strip()
                scene_label = scene_label_full.split('-')[0].strip() if scene_label_full else ""

                original_query = data.get("初始查询", "").strip()
                if not original_query:
                    continue

                for doc in data.get("生成文档", []):
                    content = doc.get("内容", "").strip()
                    label = doc.get("标签")

                    if not content or label not in [0, 1]:
                        continue

                    rel_record = {
                        "query": original_query,
                        "doc": content,
                        "label": label,
                        "scene_label": scene_label
                    }

                    if label == 0:
                        negative_docs_set.add(content)
                        f_neg_rel.write(json.dumps(rel_record, ensure_ascii=False) + '\n')
                        neg_rel_count += 1
                    elif label == 1:
                        positive_docs_set.add(content)
                        f_pos_rel.write(json.dumps(rel_record, ensure_ascii=False) + '\n')
                        pos_rel_count += 1

            except json.JSONDecodeError:
                continue

    print(f"\n相关性标签集生成完毕！")
    print(f"-> 负相关记录数: {neg_rel_count}")
    print(f"-> 正相关记录数: {pos_rel_count}")

    print(f"\n开始写入去重后的文档集 (Document Set)...")

    with open(neg_doc_file, 'w', encoding='utf-8') as f_neg_doc:
        for doc_content in tqdm(negative_docs_set, desc="写入负文档集"):
            f_neg_doc.write(json.dumps({"doc": doc_content}, ensure_ascii=False) + '\n')

    with open(pos_doc_file, 'w', encoding='utf-8') as f_pos_doc:
        for doc_content in tqdm(positive_docs_set, desc="写入正文档集"):
            f_pos_doc.write(json.dumps({"doc": doc_content}, ensure_ascii=False) + '\n')

    print(f"\n文档集生成完毕！")
    print(f"-> 独立负文档数 (去重后): {len(negative_docs_set)}")
    print(f"-> 独立正文档数 (去重后): {len(positive_docs_set)}")
    print("\n所有文件已成功生成！")


if __name__ == "__main__":
    generate_datasets()
