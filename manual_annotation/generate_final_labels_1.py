import pandas as pd
import json
import collections
import os


def get_majority_vote(labels):
    """计算多数票（Majority Vote）"""
    count = collections.Counter(labels)
    return count.most_common(1)[0][0]


def main():
    print("开始执行多专家交叉标注对齐与投票程序...\n")

    file_fq = "../data/200_MedBench_Expert_Annotation_Task_Formatted-doctor1.xlsx"
    file_dyz = "../data/200_MedBench_Expert_Annotation_Task_Formatted-doctor2.xlsx"
    file_xsm = "../data/200_MedBench_Expert_Annotation_Task_Formatted-doctor3.xlsx"
    output_file = "200_MedBench_Expert_Annotation_Task_Formatted_Final.jsonl"

    QUERY_COL = "Query (问题)"
    ID_COL = "Doc ID (编号)"
    DOC_COL = "Content (文档内容)"
    LABEL_COL = "Label (0-不相关, 1-相关)"

    for f in [file_fq, file_dyz, file_xsm]:
        if not os.path.exists(f):
            print(f" 严重错误: 找不到文件 [{f}]")
            return

    print(" 正在读取并智能清洗三位专家的 Excel 标注文件...")
    try:
        df_fq = pd.read_excel(file_fq)
        df_dyz = pd.read_excel(file_dyz)
        df_xsm = pd.read_excel(file_xsm)
    except Exception as e:
        print(f" 读取 Excel 失败: {e}")
        return

    for df in [df_fq, df_dyz, df_xsm]:
        df.columns = df.columns.str.strip()

    try:
        df_fq = df_fq[[QUERY_COL, ID_COL, DOC_COL, LABEL_COL]].rename(columns={LABEL_COL: 'label_fq'})
        df_dyz = df_dyz[[QUERY_COL, ID_COL, LABEL_COL]].rename(columns={LABEL_COL: 'label_dyz'})
        df_xsm = df_xsm[[QUERY_COL, ID_COL, LABEL_COL]].rename(columns={LABEL_COL: 'label_xsm'})
    except KeyError as e:
        print(f"找不到基础列名: {e}")
        print(f"当前文件的真实列名长这样：{df_fq.columns.tolist()}")
        return

    print("正在根据 [Query (问题)] 和 [Doc ID (编号)] 严格对齐三份标注数据...")
    merged_df = pd.merge(df_fq, df_dyz, on=[QUERY_COL, ID_COL], how='inner')
    merged_df = pd.merge(merged_df, df_xsm, on=[QUERY_COL, ID_COL], how='inner')

    if len(merged_df) != 200:
        print(f"警告: 对齐后的数据条数为 {len(merged_df)} 条，并非预期的 200 条。")
    else:
        print("数据对齐 (共 200 条)！")

    print("正在进行多数票决算 (Majority Vote)...")
    final_records = []

    for index, row in merged_df.iterrows():
        labels = [row['label_fq'], row['label_dyz'], row['label_xsm']]
        majority_label = get_majority_vote(labels)

        final_records.append({
            "query": row[QUERY_COL],
            "doc": row[DOC_COL],
            "label": majority_label
        })

    print(f"正在生成最终文件: {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in final_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"\n 恭喜处理完成。多专家标注一致性结果已成功保存至 [{output_file}]")
    except Exception as e:
        print(f"写入文件失败: {e}")


if __name__ == "__main__":
    main()
