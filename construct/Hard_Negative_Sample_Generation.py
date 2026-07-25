import json
import re
from tqdm import tqdm

prompt_file = 'prompt.txt'
input_file = 'title_content_merge.jsonl'
output_file = 'output_prompts.jsonl'

with open(prompt_file, 'r', encoding='utf-8') as f:
    template = f.read()

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
except FileNotFoundError:
    print(f"错误: 找不到输入文件 {input_file}")
    exit(1)

print(f"开始生成 Prompt，共计 {total_lines} 条数据...")
with open(input_file, 'r', encoding='utf-8') as jsonl_file, \
        open(output_file, 'w', encoding='utf-8') as out_file:
    for line in tqdm(jsonl_file, total=total_lines, desc="合并进度", unit="条"):
        try:
            data = json.loads(line.strip())
            content = data.get('内容', '')

            parts = re.split(r'(正文档：\n\{).*?(\}\n输出的内容：)', template, flags=re.DOTALL)

            if len(parts) >= 4:     
                new_prompt = parts[0] + parts[1] + content + parts[2] + parts[3]
            else:       
                print("\n警告：未匹配到目标结构，回退到字符串替换模式。")
                fallback_placeholder = "{发育性髋关节发育不良（DDH）的研究进展"
                new_prompt = template.replace(fallback_placeholder, "{" + content)

            out_file.write(json.dumps({"query": new_prompt}, ensure_ascii=False) + '\n')

        except json.JSONDecodeError:
            tqdm.write("发现格式错误的 JSON 行，已跳过。")
            continue

print(f"\n生成完成！结果已保存至: {output_file}")
