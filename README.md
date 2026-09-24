# 🏥 MedBench: A Comprehensive Benchmark for Medical Retrieval-Augmented Generation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Conference](https://img.shields.io/badge/KDD-2027-red.svg)]()

> **📢 最新动态 (News):** 我们的论文《MedBench: A Comprehensive Benchmark for Medical Retrieval-Augmented Generation》已提交至 **KDD 2027**。

## 💡 简介 (Introduction)
现有的通用检索基准（如 MTEB、BEIR）往往依赖浅层词汇匹配，难以真实反映检索模型在复杂医疗场景下的表现。**MedBench** 是一个专为医疗 RAG（检索增强生成）管道中**检索组件**设计的综合性评估基准。

本项目的**三大核心贡献 (Our Contributions)** 如下：

1. **构建高保真、多场景的医学基准 (High-fidelity, Multi-scenario Benchmark)**
   MedBench 真实反映了实际的临床工作流程。该基准全面覆盖了诊疗 (Diagnosis & Treatment)、医疗保健 (Healthcare)、就医 (Medical Visit)、疾病预防 (Prevention)、孕产 (Maternity) 和衍生服务 (Derivative Services) 等六大核心场景，弥补了现有基准在真实业务场景覆盖上的不足。

2. **开发端到端的自动化构建管道 (End-to-end Automated Construction Pipeline)**
   我们提出了一套包含语料库准备、数据集生成与严格质量控制的完整构建管道。
3. **构建独立的真实世界评估集 (Independent Real-world Evaluation Set)**
   为了验证基准测试结果能否可靠地反映模型在实际应用中的性能，我们利用从在线平台收集的真实数据构建了一个完全独立的真实世界评估集。这彻底消除了数据泄露的风险，并有效揭示了通用检索模型在医疗垂直领域中的“浅层匹配错觉”。

## 📂 仓库结构 (Repository Structure)

```text
MedBench/
├── construct/           # 数据集构建脚本 (例如: 合并相关性标签集合)
├── evaluate/            # 纯结构化的 JSONL 评测脚本 (NDCG, Recall 计算)
├── manual_annotation/   # 多专家交叉标注对齐与投票程序 (一致性检验)
└── README.md            # 项目说明文档

```
## 🏆 盲测打榜指南 (Submit to Leaderboard)

为了绝对保证测试集的纯洁性，防止数据泄露与定向过拟合 (Data Contamination)，MedBench 采用严苛的**纯黑盒盲测机制 (Blind Test)**。真实的 336 万篇医学候选文档和 1970 个查询集不予公开。

我们欢迎社区提交模型参与评测，请按照以下步骤操作：

### 1. 本地开发与调试
我们在 `data/` 目录下提供了微型的假数据（Dummy Data）：
* `dummy_corpus.jsonl`
* `dummy_queries.jsonl`

它们的 JSON 字段格式与服务器上的绝密真实数据**完全一致**。请使用这些数据在您的本地调通代码。

### 2. 封装推理代码
进入 `submission_template/` 目录，我们为您提供了标准的启动包：
* **`run_inference.py`**: 请在该脚本中加载您的模型（Embedding/Reranker）。您的脚本必须能接收 `--corpus_path`, `--query_path` 和 `--output_path` 这三个参数，并按规范输出预测结果。
* **`Dockerfile`**: 请在此配置您的运行环境。

### 3. 构建并推送镜像
在本地测试无误后，将您的代码构建为 Docker 镜像，并推送到公开的镜像仓库（如 Docker Hub）或提供私有仓库的拉取权限。
```bash
docker build -t your_dockerhub_name/medbench_model:v1 .
docker push your_dockerhub_name/medbench_model:v1
```
### 4. 提交评测申请
请在 GitHub 提交一个 Issue（标题格式：[Submission] 您的机构/团队名 - 模型名），或发送邮件至作者邮箱，附上您的 Docker 镜像拉取地址。
我们的后台自动化沙箱引擎（完全物理断网运行）将拉取您的镜像，计算相关指标。

## 📝 引用 (Citation)

如果您在研究中使用了 MedBench 的代码或数据集，请引用我们的论文：

```bibtex
@inproceedings{ma2026medbench,
  title={MedBench: A Comprehensive Benchmark for Medical Retrieval-Augmented Generation},
  author={Ma, Haiping and Zhou, Fang and Zhang, Yiyu and Hu, Jiaxue and Ma, Jun and Zhang, Xingyi},
  booktitle={Proceedings of the 33nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year={2027}
}

## 🏆 排行榜 (Leaderboard)

以下是9款主流检索模型在 MedBench 医疗语料库下的零样本（Zero-shot）检索性能表现。为提供更全面的横向对比，榜单同时收录了各模型在 AIR-Bench 24.05 (Healthcare) 与 MTEB (Healthcare) 上的基准成绩。

**主榜单按 MedBench 的 nDCG@10(%) 指标降序排列：**

| 排名 | 模型名称 (Model) | 架构分类 (Category) | 参数量 (Size) | MedBench nDCG@10(%) ⬇️ | AIR-Bench 24.05 nDCG@10(%) | MTEB (Healthcare) nDCG@10(%) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| 🥇 1 | **multilingual-e5-large-instruct** | Large-size | 0.6B | **48.40** | 39.76 | 61.14 |
| 🥈 2 | **inf-retriever-v1-1.5b** | LLM-based | 1.5B | 41.60 | 40.35 | 65.02 |
| 🥉 3 | **multilingual-e5-small** | Large-size | 0.1B | 40.34 | 28.97 | 56.75 |
| 4 | gte-Qwen2-1.5B-instruct | LLM-based | 1.5B | 39.52 | 39.13 | 67.59 |
| 5 | jina-embeddings-v3 | Large-size | 0.6B | 39.35 | 38.92 | 62.25 |
| 6 | gte-Qwen2-7B-instruct | LLM-based | 7B | 33.70 | 38.66 | 68.09 |
| 7 | inf-retriever-v1 | LLM-based | 7B | 24.23 | **41.82** | 68.80 |
| 8 | gte-multilingual-base | Large-size | 0.3B | 20.60 | 37.94 | 58.36 |
| 9 | gte-large-zh | Large-size | 0.3B | 10.71 | 29.84 | **86.46** |

> 📌 **核心洞察：** 
> 评测结果显示，通用医疗领域的检索表现与高难度专业医疗检索之间存在显著的领域壁垒。例如，`gte-large-zh` 在 MTEB (Healthcare) 中取得了 86.46% 的最高分，但在 MedBench 中仅获得 10.71% 的底层成绩；而 `multilingual-e5-small` 在其他双榜中均垫底（排名第9），却在 MedBench 冲入了前三名。这充分印证了构建 MedBench 这一垂直高难度基准的必要性。
