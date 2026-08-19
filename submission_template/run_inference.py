import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ================= 工具函数 =================
def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    【核心逻辑 1：精准掩码池化】
    原生剔除 Padding 补齐字符对池化向量的污染，比高度封装的库更能保证向量纯净度。
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# ================= 主程序 =================
def main():
    parser = argparse.ArgumentParser(description="MedBench 官方 E5 Baseline 推理模板")
    parser.add_argument("--corpus_path", type=str, required=True, help="底库文件路径")
    parser.add_argument("--query_path", type=str, required=True, help="查询文件路径")
    parser.add_argument("--output_path", type=str, required=True, help="预测结果输出路径")
    args = parser.parse_args()

    # ---------------- 1. 原生加载模型 ----------------
    print("🚀 1. 正在使用原生 HuggingFace 接口加载模型...", flush=True)
    # 沙箱环境中的模型挂载固定路径
    model_path = '/app/model' 
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # 开启半精度加速
        attn_implementation="eager" # 确保注意力计算的兼容性
    )
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # ---------------- 2. 读取数据 ----------------
    print(f"📖 2. 开始读取海量文档与查询数据...", flush=True)
    raw_corpus_docs = []
    with open(args.corpus_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取 Corpus"):
            item = json.loads(line)
            raw_corpus_docs.append(item["doc"])

    raw_queries = []
    with open(args.query_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取 Queries"):
            item = json.loads(line)
            raw_queries.append(item["query"])

    # ---------------- 3. 特征编码 ----------------
    print("🧠 3. 开始向量化与检索 (启用前缀绑定 & 掩码池化)...", flush=True)
    batch_size = 256  # 可根据显存大小(如24G)调整为 512
    max_len = 512
    
    # 3.1 编码 Corpus
    doc_embs_list = []
    for i in tqdm(range(0, len(raw_corpus_docs), batch_size), desc="Encoding Passages"):
        # 【核心逻辑 2：文档前缀绑定】
        batch_texts = [f"passage: {doc}" for doc in raw_corpus_docs[i: i + batch_size]]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1) # L2 归一化
        doc_embs_list.append(embeddings.cpu().float())
        
    doc_embs = torch.cat(doc_embs_list, dim=0)
    
    # 3.2 编码 Queries
    query_embs_list = []
    for i in tqdm(range(0, len(raw_queries), batch_size), desc="Encoding Queries"):
        # 【核心逻辑 3：查询前缀绑定】
        batch_texts = [f"query: {q}" for q in raw_queries[i: i + batch_size]]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        query_embs_list.append(embeddings.cpu().float())
        
    query_embs = torch.cat(query_embs_list, dim=0)

    # ---------------- 4. 相似度计算与文件写入 ----------------
    print("💾 4. 计算相似度并写入纯文本对齐结果...", flush=True)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 将 doc_embs 转移到 GPU 进行极速矩阵相乘 (如果显存允许)
    if torch.cuda.is_available():
        doc_embs = doc_embs.cuda()
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for i, q_text in enumerate(tqdm(raw_queries, desc="计算 Top-10")):
            # 逐个 Query 计算，防止 OOM
            q_emb = query_embs[i].unsqueeze(0)
            if torch.cuda.is_available():
                q_emb = q_emb.cuda()
                
            scores = (q_emb @ doc_embs.T).squeeze(0)
            top_scores, top_indices = torch.topk(scores, k=10)
            
            top_scores = top_scores.cpu().numpy()
            top_indices = top_indices.cpu().numpy()
            
            for rank, idx in enumerate(top_indices):
                res = {
                    "query": raw_queries[i],
                    "doc": raw_corpus_docs[idx],
                    "score": float(top_scores[rank])
                }
                f.write(json.dumps(res, ensure_ascii=False) + '\n')

    print("✅ 官方基线推理全部完成！", flush=True)

if __name__ == "__main__":
    main()
