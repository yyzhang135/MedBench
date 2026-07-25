# 🏥 MedBench: A Comprehensive Benchmark for Medical Retrieval-Augmented Generation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Data](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-yellow)](https://huggingface.co/yyzhang135)
[![Conference](https://img.shields.io/badge/KDD-2026-red.svg)]()

> **📢 最新动态 (News):** 我们的论文《MedBench: A Comprehensive Benchmark for Medical Retrieval-Augmented Generation》已提交至 **KDD 2026**。

## 💡 简介 (Introduction)
现有的通用检索基准（如 MTEB、BEIR）往往依赖浅层词汇匹配，难以真实反映检索模型在复杂医疗场景下的表现。**MedBench** 是一个专为医疗 RAG（检索增强生成）管道中**检索组件**设计的综合性评估基准。

本基准真实还原了临床与日常医疗工作流，核心特性包括：
* **六大核心医疗场景**：全面覆盖诊疗 (Diagnosis & Treatment)、医疗保健 (Healthcare)、就医 (Medical Visit)、疾病预防 (Prevention)、孕产 (Maternity) 和衍生服务 (Derivative Services)。
* **高度逼真的困难负样本 (Hard Negatives)**：包含意图偏移、条件不符、概念替换等 7 种维度的欺骗性负样本，精准鉴别模型的深度语义推理能力。
* **严格的共识机制**：数据构建阶段采用严格的质量控制过滤，模型与人类专家交叉标注的共识阈值放宽至 4 票，以在数据规模与极高质量之间取得最佳平衡。

## 📂 仓库结构 (Repository Structure)

本仓库秉承**“代码与数据解耦”**的原则，仅包含轻量级的核心执行脚本。所有的 JSONL 格式结构化数据集均托管在 Hugging Face。

```text
MedBench/
├── construct/           # 数据集构建脚本 (例如: 合并相关性标签集合)
├── evaluate/            # 纯结构化的 JSONL 评测脚本 (NDCG, Recall 计算)
├── manual_annotation/   # 多专家交叉标注对齐与投票程序 (一致性检验)
└── README.md            # 项目说明文档
