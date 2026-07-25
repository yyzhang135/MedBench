import json
import os


def main():
    input_file = "200_MedBench_Expert_Annotation_Task_Formatted_Final.jsonl"
    output_file = "200_LLM_Prompts.jsonl"
    batch_size = 10

    print(f"正在读取数据文件: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    if len(records) != 200:
        print(f" 警告: 读取到的数据条数为 {len(records)}，并非预期的 200 条。")

    PROMPT_TEMPLATE = """你是一个专业的医学搜索相关性评估专家。下面共有 10 组数据（附带 inner_idx），请分别判断每组中 "query" 和 "doc" 的相关性。

【评判标准】
符合以下任意一种情况即可标记为 1（相关）：
1. 精准解答：生成文档的前提、主体、核心意图与查询内容能够直接支撑回答，且给出了明确、正面的解答。
2. 严格同义：生成文档回答了核心意图，主体使用了医学上完全等价的专业同义词（如“高血压”与“原发性高血压”），且不遗漏查询中的任何限制条件（如特定人群、特定数值）。
3. 关键支持信息：虽未直接给出结论，但提供可直接推导答案的核心医学依据。

符合以下任意一种情况即标记为 0（不相关）：
1. 主体/意图偏移：描述的事物或核心意图与查询内容不同（如：问“高血压”，回“糖尿病”；问“病因”，回“价格”）。
2. 条件冲突/偷换概念：生成文档的内容与查询内容的特定限制条件发生冲突，或偷换了看似相近但医学本质不同的概念（如：问“儿童”，回“成人”；问“乙肝患者”，回“乙肝携带者”）。
3. 模棱两可/缺乏定论：生成文档使用了不确定的语言、反问句，或仅讨论“如何寻找答案”而未提供实质性结论，导致用户无法直接应用该信息。
4. 范围不匹配/避重就轻：对于多维度的查询，生成文档仅回答了极少部分遗漏核心；或用宽泛的大领域通用知识，去回答特定小领域的精确问题（如：问“产后腹直肌恢复”，回“产后一般生理康复”），缺乏针对性。

【输出要求】
必须直接输出 10 行 JSON 对象，每行一个。
不要使用任何方括号 `[]` 包含它们，也不要输出任何开场白、解释或 markdown 代码块标记。
格式必须严格如下（注意替换 inner_idx 为给定的序号）：
{"inner_idx": 序号, "label": 0或1}
{"inner_idx": 序号, "label": 0或1}
...
注意：输出中不需要重复 query 和 doc 的内容，仅保留 inner_idx 和 label。

【待处理数据】
<data>
{batch_data_json}
</data>"""

    print(f"正在将数据按每组 {batch_size} 条切分，并整合至 {output_file} ...")

    total_batches = (len(records) + batch_size - 1) // batch_size

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_records = records[start_idx:end_idx]

            formatted_batch = []
            for local_idx, record in enumerate(batch_records):
                global_idx = start_idx + local_idx
                formatted_batch.append({
                    "inner_idx": global_idx,
                    "query": record["query"],
                    "doc": record["doc"]
                })

            batch_json_str = json.dumps(formatted_batch, ensure_ascii=False, indent=2)

            final_prompt = PROMPT_TEMPLATE.replace("{batch_data_json}", batch_json_str)

            jsonl_record = {
                "prompt": final_prompt
            }
            out_f.write(json.dumps(jsonl_record, ensure_ascii=False) + '\n')

    print(f"已生成 {total_batches} 个 Batch 的 Prompt，全部保存在 [{output_file}] 文件中。")


if __name__ == "__main__":
    main()
