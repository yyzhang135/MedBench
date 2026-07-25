import json
import random
from pathlib import Path
from tqdm import tqdm

files_config = {
    "magazine": {
        "path": Path(
            "magazine_filtered_0.6.jsonl"),
        "total_lines": 
    },
    "guide": {
        "path": Path(
            "guide_filtered_0.6.jsonl"),
        "total_lines": 
    },
    "textbook": {
        "path": Path(
            "textbook_filtered_0.6.jsonl"),
        "total_lines": 
    },
}

TOTAL_TARGET = 
output_file = Path("proportional_merged.jsonl")


def calculate_proportions(config, target_total):

    grand_total = sum(info["total_lines"] for info in config.values())
    sampling_plan = {}
    current_sum = 0

    items = list(config.items())
    for i, (name, info) in enumerate(items):
        if i < len(items) - 1:
            count = int((info["total_lines"] / grand_total) * target_total)
            sampling_plan[name] = count
            current_sum += count
        else:
            sampling_plan[name] = target_total - current_sum

    return sampling_plan


def random_sample_jsonl(file_path: Path, n: int, total: int, seed=42):
    random.seed(seed)
    samples = []

    if n >= total:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.readlines()

    with open(file_path, "r", encoding="utf-8") as f:
        pbar = tqdm(enumerate(f), total=total, desc=f"采样 {file_path.name[:20]}...", leave=False)
        for i, line in pbar:
            if i < n:
                samples.append(line)
            else:
                if random.random() < n / (i + 1):
                    samples[random.randint(0, n - 1)] = line
    return samples

plan = calculate_proportions(files_config, TOTAL_TARGET)
print("--- 抽样计划 ---")
for name, count in plan.items():
    percentage = (count / TOTAL_TARGET) * 100
    print(f"{name}: 抽取 {count} 条 ({percentage:.2f}%)")
print("----------------\n")

all_samples = []
for name, info in files_config.items():
    path = info["path"]
    target_n = plan[name]

    if not path.exists():
        print(f"警告：文件不存在，跳过 {path}")
        continue

    samples = random_sample_jsonl(path, target_n, info["total_lines"])
    all_samples.extend(samples)

print(f"\n正在对 {len(all_samples)} 条数据进行混洗...")
random.shuffle(all_samples)

output_file.parent.mkdir(parents=True, exist_ok=True)
print(f"正在写入最终文件 {output_file} ...")
with open(output_file, "w", encoding="utf-8") as f:
    for line in tqdm(all_samples, desc="写入进度"):
        f.write(line)

print(f"\n任务完成！")
print(f"总计写入: {len(all_samples):,} 条")
