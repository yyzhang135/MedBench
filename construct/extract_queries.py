import json
import os
from tqdm import tqdm

input_file = 'LLM_output.jsonl'
output_file = 'queries_set.jsonl'


def extract_queries():
    unique_queries = set()

    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        return

    print(f"开始从 {input_file} 提取查询和场景标签...")
    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin, desc="解析进度"):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                scene_label_full = data.get("场景标签", "").strip()
                scene_label = scene_label_full.split('-')[0].strip() if scene_label_full else ""

                initial_query = data.get("初始查询", "").strip()
                if initial_query:
                    unique_queries.add((initial_query, scene_label))
                for q in data.get("改写后的查询集", []):
                    query_text = q.get("查询", "").strip()
                    if query_text:
                        unique_queries.add((query_text, scene_label))

            except json.JSONDecodeError:
                continue

    print(f"\n提取完毕！共获得 {len(unique_queries)} 条不重复的 Query。")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"正在写入 {output_file} ...")
    with open(output_file, 'w', encoding='utf-8') as fout:
        for query, scene_label in tqdm(unique_queries, desc="写入进度"):
            # 采用标准的 JSONL 格式 {"query": "...", "scene_label": "..."}
            row = {
                "query": query,
                "scene_label": scene_label
            }
            fout.write(json.dumps(row, ensure_ascii=False) + '\n')

    print("\n提取任务完成")


if __name__ == "__main__":
    extract_queries()
