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

CACHE_FILE = os.path.join(OUTPUT_DIR, "corpus_embeddings_gte_multi_base.npy")
DETAIL_RESULT_FILE = os.path.join(OUTPUT_DIR, "retrieval_details.json")
SUMMARY_RESULT_FILE = os.path.join(OUTPUT_DIR, "evaluation_summary.txt")

MODEL_PATH = " "

def get_text_hash(text):
    clean_text = str(text).strip().replace('\r', '').replace('\n', '')
    return hashlib.md5(clean_text.encode('utf-8')).hexdigest()

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

class MedBenchEvaluator:
    def __init__(self):
        self.corpus = []
        self.queries = []
        self.qrels = {}
        self.neg_map = {}

    def load_data(self):
        print(f">>> 1. 加载数据...")

        with open(FILES["corpus"], 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    content = item.get("doc", "")
                    if content: self.corpus.append(content)
                except:
                    continue
        print(f"   语料库文档总数: {len(self.corpus)}")

        with open(FILES["queries"], 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    q = item.get("query", "")
                    if q: self.queries.append({"text": q, "id": get_text_hash(q)})
                except:
                    continue
        print(f"   查询集数量: {len(self.queries)}")

        with open(FILES["qrels"], 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    q_hash = get_text_hash(item.get("query", ""))
                    d_hash = get_text_hash(item.get("doc", ""))
                    if q_hash not in self.qrels: self.qrels[q_hash] = {}
                    self.qrels[q_hash][d_hash] = int(item.get("label", 0))
                except:
                    continue

        if os.path.exists(FILES["neg_qrels"]):
            with open(FILES["neg_qrels"], 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if item.get("label", 0) == 0:
                            q_hash = get_text_hash(item.get("query", ""))
                            d_hash = get_text_hash(item.get("doc", ""))
                            if q_hash not in self.neg_map: self.neg_map[q_hash] = {}
                            self.neg_map[q_hash][d_hash] = {
                                "hard_negative_category": item.get("hard_negative_category", 0),
                                "hard_negative_thought": item.get("hard_negative_thought", "")
                            }
                    except:
                        continue

    def compute_metrics(self, retrieved_ids, true_map, neg_dict, k=10):
        retrieved_ids = retrieved_ids[:k]
        dcg, idcg = 0.0, 0.0
        num_rel = sum([1 for s in true_map.values() if s > 0])
        hits, mrr, hard_neg_count = 0, 0.0, 0

        if len(retrieved_ids) == 0: return 0.0, 0.0, 0.0, 0.0, 0.0

        for i, doc_id in enumerate(retrieved_ids):
            rel = true_map.get(doc_id, 0)
            if rel > 0:
                dcg += 1.0 / np.log2(i + 2)
                hits += 1
                if mrr == 0.0: mrr = 1.0 / (i + 1)
            if doc_id in neg_dict:
                hard_neg_count += 1

        if num_rel > 0:
            for i in range(min(num_rel, k)): idcg += 1.0 / np.log2(i + 2)

        ndcg = dcg / idcg if idcg > 0 else 0.0
        recall = hits / num_rel if num_rel > 0 else 0.0
        precision = hits / len(retrieved_ids)
        hn_ratio = hard_neg_count / len(retrieved_ids)
        return precision, recall, ndcg, mrr, hn_ratio


class GTEMultiBaseRetriever:
    def __init__(self, corpus_list, batch_size=128):
        print(f"\n正在加载模型: {MODEL_PATH} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)

        if torch.cuda.is_available():
            self.model.cuda()
            self.model = self.model.half()

        self.model.eval()
        self.batch_size = batch_size
        self.corpus_docs = corpus_list

        if os.path.exists(CACHE_FILE):
            print(f"加载语料向量缓存...")
            self.corpus_embeddings = np.load(CACHE_FILE)
            if len(self.corpus_docs) != self.corpus_embeddings.shape[0]:
                self.corpus_embeddings = self._encode_corpus()
        else:
            self.corpus_embeddings = self._encode_corpus()

    def _encode_corpus(self):
        print(f"正在编码语料库 (共 {len(self.corpus_docs)} 条)...")
        dim = 768
        final_embeddings = np.zeros((len(self.corpus_docs), dim), dtype=np.float16)

        for i in tqdm(range(0, len(self.corpus_docs), self.batch_size), desc="Encoding"):
            batch_texts = self.corpus_docs[i: i + self.batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=1024)

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = cls_pooling(outputs)
                embeddings = F.normalize(embeddings, p=2, dim=1)

            final_embeddings[i: i + len(batch_texts)] = embeddings.cpu().float().numpy().astype(np.float16)

        np.save(CACHE_FILE, final_embeddings)
        return final_embeddings

    def _encode_queries(self, queries):
        all_embeddings = []
        for i in tqdm(range(0, len(queries), self.batch_size), desc="Query Encoding"):
            inputs = self.tokenizer(queries[i: i + self.batch_size], padding=True, truncation=True, return_tensors='pt',
                                    max_length=1024)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = cls_pooling(outputs)
                embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().float().numpy())
        return np.concatenate(all_embeddings, axis=0)

if __name__ == "__main__":
    evaluator = MedBenchEvaluator()
    evaluator.load_data()

    gte_model = GTEMultiBaseRetriever(evaluator.corpus, batch_size=128)

    print("\n编码查询集...")
    query_texts = [q["text"] for q in evaluator.queries]
    q_embeddings = gte_model._encode_queries(query_texts)

    precision_scores, recall_scores, ndcg_scores, mrr_scores, hn_ratios = [], [], [], [], []
    top_k = 10
    detailed_logs = []

    print(f"\n检索评估...")
    corpus_embeddings_f32 = gte_model.corpus_embeddings.astype(np.float32)

    for i, query in enumerate(tqdm(evaluator.queries, desc="Evaluating")):
        qid = query["id"]
        if qid not in evaluator.qrels: continue

        q_emb = q_embeddings[i].astype(np.float32)
        scores = q_emb @ corpus_embeddings_f32.T

        top_k_indices_unordered = np.argpartition(-scores, top_k)[:top_k]
        top_k_indices = top_k_indices_unordered[np.argsort(-scores[top_k_indices_unordered])]

        retrieved_ids = [get_text_hash(gte_model.corpus_docs[idx]) for idx in top_k_indices]
        current_neg_dict = evaluator.neg_map.get(qid, {})

        metrics = evaluator.compute_metrics(retrieved_ids, evaluator.qrels[qid], current_neg_dict, k=top_k)

        precision_scores.append(metrics[0]);
        recall_scores.append(metrics[1])
        ndcg_scores.append(metrics[2]);
        mrr_scores.append(metrics[3])
        hn_ratios.append(metrics[4])

        top_10_log_list = []
        for rank, idx in enumerate(top_k_indices):
            d_hash = get_text_hash(gte_model.corpus_docs[idx])
            is_hn = d_hash in current_neg_dict

            doc_log = {
                "rank": rank + 1,
                "score": round(float(scores[idx]), 4),
                "is_hard_negative": is_hn,
                "content": gte_model.corpus_docs[idx][:200]
            }
            if is_hn:
                hn_metadata = current_neg_dict[d_hash]
                doc_log["hard_negative_category"] = hn_metadata.get("hard_negative_category", 0)
                doc_log["hard_negative_thought"] = hn_metadata.get("hard_negative_thought", "")
            top_10_log_list.append(doc_log)

        detailed_logs.append({
            "query": query["text"],
            "metrics": {
                "precision": round(metrics[0], 4),
                "recall": round(metrics[1], 4),
                "ndcg": round(metrics[2], 4),
                "mrr": round(metrics[3], 4),
                "hard_negative_ratio": round(metrics[4], 4)
            },
            "top_10_retrieved": top_10_log_list
        })

    mean_precision = np.mean(precision_scores) if precision_scores else 0.0
    mean_recall = np.mean(recall_scores) if recall_scores else 0.0
    mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    mean_mrr = np.mean(mrr_scores) if mrr_scores else 0.0
    mean_hn_ratio = np.mean(hn_ratios) if hn_ratios else 0.0
    evaluated_count = len(ndcg_scores)

    print("\n" + "=" * 60)
    print(f"GTE-Multilingual-Base Results ( {evaluated_count} queries @ Top-{top_k}):")
    print(f"   Precision (精确率)        : {mean_precision:.4f}")
    print(f"   Recall    (召回率)        : {mean_recall:.4f}")
    print(f"   NDCG      (排序质量)      : {mean_ndcg:.4f}")
    print(f"   MRR       (首位相关率)    : {mean_mrr:.4f}")
    print("=" * 60)

    summary_log = f"Model: GTE-Multilingual-Base | Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Precision: {mean_precision:.4f} | Recall: {mean_recall:.4f} | NDCG: {mean_ndcg:.4f} | MRR: {mean_mrr:.4f} | HN_Ratio: {mean_hn_ratio:.4f}\n"
    with open(SUMMARY_RESULT_FILE, "a", encoding="utf-8") as f:
        f.write(summary_log)

    with open(DETAIL_RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(detailed_logs, f, ensure_ascii=False, indent=4)

    print(f"GTE-Multilingual-Base 评测完成！结果已保存。")
