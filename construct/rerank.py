# -*- coding: utf-8 -*-
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = " "
 
RETRIEVAL_RESULT_PATH = os.path.join(BASE_DIR, "1000_retrieval_with.jsonl")

OUTPUT_PATH = os.path.join(BASE_DIR,"voted_pre_labeled_candidates.jsonl")

MODELS_CONFIG = [
    {
        "name": "bge-reranker-large",
        "path": os.path.join(BASE_DIR, "bge-reranker-large")
    },
    {
        "name": "gte-multilingual-reranker-base",
        "path": os.path.join(BASE_DIR,"gte-multilingual-reranker-base")
    },
    {
        "name": "jina-reranker-v2-base-multilingual",
        "path": os.path.join(BASE_DIR,"jina-reranker-v2-base-multilingual")
    }
]

RANK_THRESHOLD = 100

MIN_VOTES = 2

INFERENCE_BATCH_SIZE = 32

def load_model_on_device(config, device_id):
    device_str = f"cuda:{device_id}"
    print(f"正在加载模型: {config['name']} -> {device_str} ...")
    path = config['path']
    if not os.path.exists(path):
        print(f"找不到模型路径: {path}")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2"
            ).to(device_str)
        except:
            model = AutoModelForSequenceClassification.from_pretrained(
                path,
                trust_remote_code=True,
                torch_dtype=torch.float16
            ).to(device_str)

        model.eval()
        return {
            "name": config["name"],
            "model": model,
            "tokenizer": tokenizer,
            "device": device_str
        }
    except Exception as e:
        print(f"Error loading {config['name']}: {e}")
        return None


def get_scores_and_ranks(model_bundle, query, docs, batch_size=INFERENCE_BATCH_SIZE):
    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    device = model_bundle["device"]

    pairs = [[query, doc] for doc in docs]
    all_scores = []

    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i: i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
            output = model(**inputs, return_dict=True)
            logits = output.logits.view(-1).float().cpu().numpy()
            all_scores.extend(logits)

    indexed_scores = list(enumerate(all_scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    doc_ranks = [0] * len(docs)
    for rank_idx, (original_idx, score) in enumerate(indexed_scores):
        doc_ranks[original_idx] = rank_idx + 1

    return doc_ranks, all_scores


def main():
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 张 GPU 可用 (已强制限制为单卡 GPU 0)")
    if num_gpus == 0:
        num_gpus = 1

    loaded_models = []
    for i, cfg in enumerate(MODELS_CONFIG):
        target_device_id = 0  
        m = load_model_on_device(cfg, target_device_id)
        if m: loaded_models.append(m)

    if len(loaded_models) < 3:
        print(f"警告: 仅加载了 {len(loaded_models)} 个模型。")

    print(f"开始集成重排序 (Sequential Ensemble Reranking on Single GPU)...")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(RETRIEVAL_RESULT_PATH, 'r', encoding='utf-8') as f_in, \
            open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc="Processing Queries (Top-1000 Full Rerank)"):
            try:
                item = json.loads(line)
            except:
                continue

            query = item.get("query", "").strip()
            retrieved_objs = item.get("retrieved_documents", [])
            if not query or not retrieved_objs: continue

            doc_contents = [d.get("doc", "").strip() for d in retrieved_objs]

            all_models_ranks = [None] * len(loaded_models)
            all_models_scores = [None] * len(loaded_models)

            for idx, m_bundle in enumerate(loaded_models):
                try:
                    ranks, scores = get_scores_and_ranks(m_bundle, query, doc_contents)
                    all_models_ranks[idx] = ranks
                    all_models_scores[idx] = scores
                except Exception as e:
                    print(f"模型 {m_bundle['name']} 推理出错: {e}")
                    all_models_ranks[idx] = [9999] * len(doc_contents)
                    all_models_scores[idx] = [-9999] * len(doc_contents)

            candidates = []
            for doc_idx, doc_content in enumerate(doc_contents):
                votes = 0
                detail_ranks = {}
                valid_scores = []

                for model_idx, ranks_list in enumerate(all_models_ranks):
                    if not ranks_list: continue
                    rank = ranks_list[doc_idx]
                    score = all_models_scores[model_idx][doc_idx]

                    model_name = loaded_models[model_idx]["name"]
                    detail_ranks[model_name] = rank
                    valid_scores.append(score)

                    if rank <= RANK_THRESHOLD:
                        votes += 1

                if votes >= MIN_VOTES:
                    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
                    candidates.append({
                        "document": doc_content,
                        "votes": votes,
                        "detail_ranks": detail_ranks,
                        "reason": f"Ranked high by {votes}/{len(loaded_models)} models",
                        "avg_rerank_score": float(avg_score)
                    })

            if candidates:
                candidates.sort(key=lambda x: x["avg_rerank_score"], reverse=True)
                f_out.write(json.dumps({
                    "query": query,
                    "total_candidates": len(candidates),
                    "candidates": candidates
                }, ensure_ascii=False) + "\n")

    print(f"完成！结果已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
