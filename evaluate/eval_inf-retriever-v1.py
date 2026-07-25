# -*- coding: utf-8 -*-
import json
import os
import hashlib
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

BASE_DIR = " "

OUTPUT_DIR = os.path.join(BASE_DIR, " ")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

DATA_DIR = os.path.join(BASE_DIR, " ")
FILES = {
    "corpus": os.path.join(DATA_DIR, "corpus.jsonl"),
    "queries": os.path.join(DATA_DIR, "queries_set.jsonl"),
    "qrels": os.path.join(DATA_DIR, "test_qrels.jsonl"),
    "neg_qrels": os.path.join(DATA_DIR, "test_qrels.jsonl")
}

CACHE_FILE = os.path.join(OUTPUT_DIR, "corpus_embeddings_inf_v1.npy")
DETAIL_RESULT_FILE = os.path.join(OUTPUT_DIR, "retrieval_details.json")
SUMMARY_RESULT_FILE = os.path.join(OUTPUT_DIR, "evaluation_summary.txt")

MODEL_PATH = " "
TASK_INSTRUCTION = "Instruct: Given a medical question, retrieve relevant documents that answer the query.\nQuery: "

def get_text_hash(text):
    return hashlib.md5(str(text).strip().encode('utf-8')).hexdigest()


def cls_pool(last_hidden_states):
    return last_hidden_states[:, 0]


def compute_metrics(retrieved_hashes, true_map, neg_dict, k=10):
    retrieved_hashes = retrieved_hashes[:k]
    dcg, idcg = 0.0, 0.0
    num_rel = sum([1 for s in true_map.values() if s > 0])
    hits, mrr, hard_neg_count = 0, 0.0, 0

    if not retrieved_hashes or num_rel == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    for i, h in enumerate(retrieved_hashes):
        rel = true_map.get(h, 0)
        if rel > 0:
            dcg += 1.0 / np.log2(i + 2)
            hits += 1
            if mrr == 0.0: mrr = 1.0 / (i + 1)
        if h in neg_dict:
            hard_neg_count += 1

    for i in range(min(num_rel, k)):
        idcg += 1.0 / np.log2(i + 2)

    ndcg = dcg / idcg if idcg > 0 else 0.0
    recall = hits / num_rel
    precision = hits / len(retrieved_hashes)
    hn_ratio = hard_neg_count / len(retrieved_hashes)
    return precision, recall, ndcg, mrr, hn_ratio


class InfRetrieverV1:

    def __init__(self, corpus_list, batch_size=2):
        print(f"\n正在加载 inf-retriever-v1 模型至 GPU (BF16)...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        self.model = AutoModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).cuda()

        self.model.eval()
        self.batch_size = batch_size
        self.corpus_docs = corpus_list

        if os.path.exists(CACHE_FILE):
            print(f"发现缓存！尝试加载: {CACHE_FILE}")
            self.corpus_embeddings = np.load(CACHE_FILE)
            if len(self.corpus_docs) != self.corpus_embeddings.shape[0]:
                print("缓存条数不匹配，将自动重新计算...")
                self.corpus_embeddings = self._encode_batch(self.corpus_docs, "Corpus")
                np.save(CACHE_FILE, self.corpus_embeddings)
        else:
            self.corpus_embeddings = self._encode_batch(self.corpus_docs, "Corpus")
            np.save(CACHE_FILE, self.corpus_embeddings)

    def _encode_batch(self, texts, desc):

        if desc == "Corpus":
            print(f"正在编码 {len(texts)} 条语料...")
        elif desc == "Queries":
            print("正在编码查询集...")

        all_embeddings = []
        max_len = 512

        for i in tqdm(range(0, len(texts), self.batch_size), desc=f"Encoding {desc}"):
            batch = texts[i: i + self.batch_size]
            if desc == "Queries":
                batch = [f"{TASK_INSTRUCTION}{t}" for t in batch]

            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors='pt').to(
                "cuda")

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = cls_pool(outputs.last_hidden_state)
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().to(torch.float16).numpy())

        return np.concatenate(all_embeddings, axis=0)


def run_eval():
    corpus = []
    with open(FILES["corpus"], 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if "doc" in item: corpus.append(item["doc"])
    print(f"   语料库： {len(corpus)} 条")

    qrels, neg_map = {}, {}
    with open(FILES["qrels"], 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            q_h, d_h = get_text_hash(item["query"]), get_text_hash(item["doc"])
            if q_h not in qrels: qrels[q_h] = {}
            qrels[q_h][d_h] = int(item.get("label", 1))
    print("    Qrels 加载完成。")

    if os.path.exists(FILES["neg_qrels"]):
        with open(FILES["neg_qrels"], 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                q_h, d_h = get_text_hash(item["query"]), get_text_hash(item["doc"])
                if q_h not in neg_map: neg_map[q_h] = {}
                neg_map[q_h][d_h] = 1

    queries_data = []
    with open(FILES["queries"], 'r', encoding='utf-8') as f:
        for line in f: queries_data.append(json.loads(line))
    print(f"   查询集： {len(queries_data)} 条")

    if not corpus:
        print("语料库为空，退出。")
        return

    engine = InfRetrieverV1(corpus, batch_size=2)

    q_texts = [q["query"] for q in queries_data]

    q_embs = engine._encode_batch(q_texts, "Queries")

    del engine.model
    torch.cuda.empty_cache()

    top_k = 10

    q_embs_np = q_embs.astype(np.float32)
    c_embs_np = engine.corpus_embeddings.astype(np.float32)

    precision_scores, recall_scores, ndcg_scores, mrr_scores, hn_ratios = [], [], [], [], []
    detailed_results = []

    for i in tqdm(range(len(queries_data)), desc="Evaluating"):
        q_text = queries_data[i]["query"]
        q_hash = get_text_hash(q_text)
        if q_hash not in qrels: continue

        scores = q_embs_np[i] @ c_embs_np.T

        top_k_indices_unordered = np.argpartition(-scores, top_k)[:top_k]
        indices = top_k_indices_unordered[np.argsort(-scores[top_k_indices_unordered])]

        retrieved_hashes = [get_text_hash(engine.corpus_docs[idx]) for idx in indices]

        p, r, n, m, hn = compute_metrics(retrieved_hashes, qrels[q_hash], neg_map.get(q_hash, {}), k=top_k)

        precision_scores.append(p)
        recall_scores.append(r)
        ndcg_scores.append(n)
        mrr_scores.append(m)
        hn_ratios.append(hn)

        detailed_results.append({
            "query": q_text,
            "retrieved_docs": [engine.corpus_docs[idx] for idx in indices],
            "metrics": {"precision": p, "recall": r, "ndcg": n, "mrr": m, "hn_ratio": hn}
        })

    with open(DETAIL_RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_ndcg = np.mean(ndcg_scores)
    mean_mrr = np.mean(mrr_scores)
    mean_hn_ratio = np.mean(hn_ratios)

    print("\n" + "=" * 60)
    print(f"inf-retriever-v1 Results (@Top-{top_k}):")
    print(f"   Precision (精确率)        : {mean_precision:.4f}")
    print(f"   Recall    (召回率)        : {mean_recall:.4f}")
    print(f"   NDCG      (排序质量)      : {mean_ndcg:.4f}")
    print(f"   MRR       (首位相关率)    : {mean_mrr:.4f}")
    print("=" * 60)

    with open(SUMMARY_RESULT_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.datetime.now()} | Model: inf-retriever-v1 | Precision: {mean_precision:.4f}, NDCG: {mean_ndcg:.4f}, Recall: {mean_recall:.4f}, MRR: {mean_mrr:.4f}, HN: {mean_hn_ratio:.4f}\n"
        )


if __name__ == "__main__":
    run_eval()
