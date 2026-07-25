import json

input_file = 'proportional_merged.jsonl'
output_file = 'title_content_merge.jsonl'

with open(output_file, 'w', encoding='utf-8') as outfile:
    with open(input_file, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                print(f"第 {i} 行为空，跳过")
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"第 {i} 行解析失败: {e}")
                print(f"行内容: {repr(line)}")
                continue
            title = data.get('title', '').strip()
            content = data.get('content', '').strip()
            merged_text = f'{title}\n{content}'.strip()
            result = {"内容": merged_text}   
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"合并完成! 已保存到 {output_file}")
