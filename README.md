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

```
## 📝 引用 (Citation)

如果您在研究中使用了 MedBench 的代码或数据集，请引用我们的论文：

```bibtex
@inproceedings{ma2026medbench,
  title={MedBench: A Comprehensive Benchmark for Medical Retrieval-Augmented Generation},
  author={Ma, Haiping and Zhou, Fang and Zhang, Yiyu and Hu, Jiaxue and Ma, Jun and Zhang, Xingyi},
  booktitle={Proceedings of the 33nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year={2027}
}
