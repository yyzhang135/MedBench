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

本仓库秉承**“代码与数据解耦”**的原则，仅包含轻量级的核心执行脚本。所有的 JSONL 格式结构化数据集均托管在 Hugging Face。

```text
MedBench/
├── construct/           # 数据集构建脚本 (例如: 合并相关性标签集合)
├── evaluate/            # 纯结构化的 JSONL 评测脚本 (NDCG, Recall 计算)
├── manual_annotation/   # 多专家交叉标注对齐与投票程序 (一致性检验)
└── README.md            # 项目说明文档
