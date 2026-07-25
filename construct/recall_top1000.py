import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import numpy as np
import torch
import sys
from sentence_transformers import SentenceTransformer, util, models
from tqdm import tqdm


BASE_DIR = " "

MODEL_PATH = os.path.join(BASE_DIR, " ")
CORPUS_EMBED_PATH = os.path.join(BASE_DIR, "corpus_embeddings_multi_e5.npy")
CORPUS_TEXT_PATH = os.path.join(BASE_DIR,"corpus.jsonl")
QUERY_PATH = os.path.join(BASE_DIR, "queries_set.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "1000_retrieval_with_rank.jsonl")

TOP_K = 1000

BATCH_SIZE = 64

QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query: "


def load_corpus_text_list(path):
    print(f"正在加载语料库文本: {path} ...")
    if not os.path.exists(path):
        print(f" 错误：语料库文件不存在 {path}")
        sys.exit(1)

    text_list = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="Reading Corpus"):
            line = line.strip()
            if not line:
                text_list.append("")
                continue
            try:
                item = json.loads(line)
                text = item.get("doc", "").strip()
                text_list.append(text)
            except json.JSONDecodeError:
                text_list.append("")
    return text_list


def load_model_manually(model_path, device):
    print(f"正在手动加载模型组件: {model_path}")
    if not os.path.exists(model_path):
        print(f" 错误：模型路径不存在 {model_path}")
        sys.exit(1)

    try:
        word_embedding_model = models.Transformer(model_path, max_seq_length=512)
        dimension = word_embedding_model.get_word_embedding_dimension()
        pooling_model = models.Pooling(dimension, pooling_mode='mean')
        normalize_model = models.Normalize()

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normalize_model], device=device)

        if "cuda" in device:
            model.half()
            print("已开启 FP16 半精度加速！")

        return model
    except Exception as e:
        print(f"手动加载失败: {e}")
        raise e


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_model_manually(MODEL_PATH, device)

    corpus_texts = load_corpus_text_list(CORPUS_TEXT_PATH)
    if len(corpus_texts) == 0:
        raise ValueError("【错误】语料库加载为空！")
    print(f"文本加载完毕，共 {len(corpus_texts)} 条。")

    run_embedding = False
    if os.path.exists(CORPUS_EMBED_PATH):
        print(f"发现向量缓存，正在加载: {CORPUS_EMBED_PATH}")
        corpus_embeddings = np.load(CORPUS_EMBED_PATH)
        if corpus_embeddings.shape[0] != len(corpus_texts):
            print(f"警告: 缓存向量数 ({corpus_embeddings.shape[0]}) 与 文本数 ({len(corpus_texts)}) 不一致！")
            print("准备重新计算向量...")
            run_embedding = True
    else:
        print(f"未发现向量缓存: {CORPUS_EMBED_PATH}")
        print("准备计算向量...")
        run_embedding = True

    if run_embedding:
        os.makedirs(os.path.dirname(CORPUS_EMBED_PATH), exist_ok=True)
        print("正在编码语料库 (极速模式启动)...")
        corpus_embeddings = model.encode(
            corpus_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        print(f"保存向量到: {CORPUS_EMBED_PATH}")
        np.save(CORPUS_EMBED_PATH, corpus_embeddings)

    queries, raw_queries = [], []
    print(f"正在加载查询: {QUERY_PATH}")
    if not os.path.exists(QUERY_PATH):
        print(f"错误：查询文件不存在 {QUERY_PATH}")
        sys.exit(1)

    with open(QUERY_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                item = json.loads(line)
                q = item.get("query", "").strip()
                if q:
                    raw_queries.append(q)
                    queries.append(QUERY_INSTRUCTION + q)
            except json.JSONDecodeError:
                continue
    print(f"成功加载查询 {len(queries)} 条。")

    print("正在编码查询...")
    q_embeds = model.encode(
        queries,
        batch_size=BATCH_SIZE,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    q_embeds = q_embeds.to(dtype=torch.float16)

    print(f"正在执行 Top-{TOP_K} 检索...")

    if isinstance(corpus_embeddings, np.ndarray):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)

    corpus_embeddings = corpus_embeddings.to(dtype=torch.float16, device=q_embeds.device)
    print("语料库载入完成，开始检索匹配")

    results = []
    search_batch_size = 2000

    for start_idx in tqdm(range(0, len(q_embeds), search_batch_size), desc="Retrieval Progress"):
        end_idx = start_idx + search_batch_size
        q_chunk = q_embeds[start_idx:end_idx]

        chunk_results = util.semantic_search(
            q_chunk,
            corpus_embeddings,
            top_k=TOP_K,
            query_chunk_size=100,
            corpus_chunk_size=100000
        )
        results.extend(chunk_results)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print(f"正在写入检索结果: {OUTPUT_PATH}")

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for q_idx, hits in enumerate(tqdm(results, desc="Writing Results")):
            retrieved_docs_list = []
            for rank_index, hit in enumerate(hits):
                c_id = hit['corpus_id']
                score = hit['score']
                txt = corpus_texts[c_id]

                doc_item = {
                    "rank": rank_index + 1,
                    "score": float(score),
                    "doc": txt
                }
                retrieved_docs_list.append(doc_item)

            out = {
                "query": raw_queries[q_idx],
                "total_hits": len(retrieved_docs_list),
                "retrieved_documents": retrieved_docs_list
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print("全部检索任务完成！")


if __name__ == "__main__":
    main()
