import json
import os
from tqdm import tqdm

medical_corpus_files = [
    'guide_filtered_0.6.jsonl',
    'magazine_filtered_0.6.jsonl',
    'textbook_filtered_0.6.jsonl'
]

llm_generated_files = [
    'negative_doc_set.jsonl',
    'positive_doc_set.jsonl'
]

output_file = './data/test_dataset/corpus.jsonl'


def generate_full_corpus():
    unique_docs = set()

    print("阶段一：处理原始医学文献库 (合并 title 与 content)...")
    for file_path in medical_corpus_files:
        if not os.path.exists(file_path):
            print(f"⚠️ 警告: 找不到文件 {file_path}，已跳过。")
            continue

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as fin:
            for line in tqdm(fin, desc=f"解析 {os.path.basename(file_path)}"):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    title = data.get("title", "").strip()
                    content = data.get("content", "").strip()

                    if title and content:
                        merged_doc = f"{title} {content}"
                    elif title:
                        merged_doc = title
                    elif content:
                        merged_doc = content
                    else:
                        continue

                    unique_docs.add(merged_doc)
                except json.JSONDecodeError:
                    continue

    print("\n阶段二：处理大模型生成的正负文档集 (提取 doc 字段)...")
    for file_path in llm_generated_files:
        if not os.path.exists(file_path):
            print(f"⚠️ 警告: 找不到文件 {file_path}，已跳过。")
            continue

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as fin:
            for line in tqdm(fin, desc=f"解析 {os.path.basename(file_path)}"):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    doc_content = data.get("doc", "").strip()
                    if doc_content:
                        unique_docs.add(doc_content)
                except json.JSONDecodeError:
                    continue

    print(f"\n所有数据提取完毕！全局去重后共获得 {len(unique_docs)} 条独立文档。")

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n阶段三：正在写入最终的全局语料库文件: {output_file} ...")
    # 写入时标准 utf-8 即可，因为进入内存的数据已经是干净的了
    with open(output_file, 'w', encoding='utf-8') as fout:
        for doc in tqdm(unique_docs, desc="写入进度"):
            record = {"doc": doc}
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')

    print("\n语料库 (corpus.jsonl)已生成")


if __name__ == "__main__":
    generate_full_corpus()
