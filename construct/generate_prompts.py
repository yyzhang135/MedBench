# -*- coding: utf-8 -*-
import json
import os
from tqdm import tqdm

BASE_DIR = " "

INPUT_PATH = os.path.join(BASE_DIR,"flattened_candidates.jsonl")

OUTPUT_PATH = os.path.join(BASE_DIR,"llm_input_prompts.jsonl")

BATCH_SIZE = 10

def construct_prompt(records):

    json_lines = []
    for record in records:
        data_part = {
            "inner_idx": record.get("inner_idx"),
            "query": record.get("query"),
            "document": record.get("document")
        }
        json_lines.append(json.dumps(data_part, ensure_ascii=False))

    combined_json_str = "\n".join(json_lines)

    prompt_text = f"""你是一个专业的搜索相关性评估专家。下面共有 {len(records)} 组数据，请分别判断每组中“query”内容和“document”内容是否相关。

符合以下任意一种情况即可标记为 1：
1.精准解答：生成文档的前提、主体、核心意图与查询内容能够直接支撑回答，且给出了明确、正面的解答。
2.严格同义：生成文档回答了核心意图，主体使用了医学上完全等价的专业同义词（如“高血压”与“原发性高血压”），且不遗漏查询中的任何限制条件（如特定人群、特定数值）。
3.关键支持信息：虽未直接给出结论，但提供可直接推导答案的核心医学依据。

符合以下任意一种情况即标记为 0：
1.主体/意图偏移：描述的事物或核心意图与查询内容不同（如：问“高血压”，回“糖尿病”；问“病因”，回“价格”）。
2.条件冲突/偷换概念：生成文档的内容与查询内容的特定限制条件发生冲突，或偷换了看似相近但医学本质不同的概念（如：问“儿童”，回“成人”；问“乙肝患者”，回“乙肝携带者”）。
3.模棱两可/缺乏定论：生成文档使用了不确定的语言、反问句，或仅讨论“如何寻找答案”而未提供实质性结论，导致用户无法直接应用该信息。
4.范围不匹配/避重就轻：对于多维度的查询，生成文档仅回答了极少部分遗漏核心；或用宽泛的大领域通用知识，去回答特定小领域的精确问题（如：问“产后腹直肌恢复”，回“产后一般生理康复”），缺乏针对性。

输出要求：
请直接输出 {len(records)} 行 JSON 对象，每行格式为：{{"query":"query的内容", "inner_idx":整型的数字 , "label":0或1}}。
不要输出任何开场白、解释或总结。

待处理数据：
{combined_json_str}"""

    return prompt_text


def main():
    print(f" 正在生成批处理 Prompt (每批 {BATCH_SIZE} 条)...")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"找不到输入文件: {INPUT_PATH}")

    count = 0
    batch = []

    with open(INPUT_PATH, 'r', encoding='utf-8') as f_in, \
            open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc="Batching Prompts"):
            try:
                record = json.loads(line)
                batch.append(record)
            except json.JSONDecodeError:
                continue

            if len(batch) == BATCH_SIZE:
                full_prompt = construct_prompt(batch)
                f_out.write(json.dumps({"query": full_prompt}, ensure_ascii=False) + "\n")
                count += 1
                batch = [] 

        if batch:
            full_prompt = construct_prompt(batch)
            f_out.write(json.dumps({"query": full_prompt}, ensure_ascii=False) + "\n")
            count += 1

    print(f"\n 处理完成！共生成了 {count} 个批处理 Prompt。")
    print(f" 结果已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
