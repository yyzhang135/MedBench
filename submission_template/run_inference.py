import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser(description="MedBench 盲测推理脚本模板")
    parser.add_argument("--corpus_path", type=str, required=True, help="候选文档库路径 (JSONL)")
    parser.add_argument("--query_path", type=str, required=True, help="查询集路径 (JSONL)")
    parser.add_argument("--output_path", type=str, required=True, help="输出预测结果路径 (JSONL)")
    args = parser.parse_args()

    print("🚀 初始化模型...")
    # TODO: 参赛者在此处加载自己的 Embedding / Reranker 模型
    
    print(f"📖 读取候选文档库: {args.corpus_path}")
    # TODO: 参赛者解析语料库，提取 doc_id 和文本
    
    print(f"❓ 读取查询集: {args.query_path}")
    # TODO: 参赛者解析查询集，提取 query_id 和文本

    print("🧠 执行检索推理...")
    # TODO: 参赛者计算相似度，获取 Top-K 结果

    print(f"💾 保存检索结果至: {args.output_path}")
    # 强制要求的输出格式示例
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        # 假设 q_001 检索到了 d_000001，相似度得分为 0.95
        dummy_result = {"query_id": "q_001", "doc_id": "d_000001", "score": 0.95}
        f.write(json.dumps(dummy_result, ensure_ascii=False) + '\n')
        
    print("✅ 推理完成！")

if __name__ == "__main__":
    main()
