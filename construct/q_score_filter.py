import json
import os
from tqdm import tqdm

input_file = r'magazine_merged.jsonl'
output_file = r'magazine_filtered_0.6.jsonl'

output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("正在统计总行数...")
with open(input_file, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

with open(input_file, 'r', encoding='utf-8') as fin, \
        open(output_file, 'w', encoding='utf-8') as fout:
    count = 0
    match_count = 0

    pbar = tqdm(fin, total=total_lines, desc="过滤进度", unit="line")

    for line in pbar:
        count += 1
        try:
            data = json.loads(line.strip())

            if data.get('q_score', 0) > 0.6:
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                match_count += 1

            if count % 1000 == 0:
                pbar.set_postfix({"已保留": match_count})

        except json.JSONDecodeError:
            pbar.write(f"跳过格式错误的行: {count}")
            continue

print(f"\n处理完成！")
print(f"总处理行数: {count}")
print(f"符合条件行数: {match_count}")
