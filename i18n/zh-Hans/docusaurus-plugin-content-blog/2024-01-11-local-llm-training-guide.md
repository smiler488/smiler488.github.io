---
slug: local-llm-training-guide-en
title: "本地大模型：运行、微调与部署的可复现指南"
description: "一套可维护的工作流：选择本地模型、用 Ollama 测试、规划参数高效微调、评估结果，并安全地对外提供服务。"
authors: [liangchao]
tags: [artificial-intelligence, machine-learning, local-ai, reproducible-research]
image: /img/blog-default.jpg
category: "AI & machine learning"
article_type: Technical guide
---

当数据必须留在受控硬件上、当离线运行很重要、或者当实验需要固定的模型与软件栈时，本地大语言模型就有了用武之地。但它们并不天然更便宜、更快或更私密：这些结果取决于模型大小、硬件、网络设置，以及服务对外暴露的方式。

本指南把三件常被混为一谈的事分开：**运行模型**、**适配模型**和**部署模型**。请从能回答你研究问题的最小那件事开始。

<!-- truncate -->

:::caution 本工作流对版本敏感

模型名称、软件包 API、CUDA 构建和硬件要求变化很快。对每个需要复现的实验，请记录模型修订版本、依赖锁定文件、操作系统、驱动、加速器、提示词模板和测试日期。下面的命令是起点，不是放之四海皆准的生产配方。

:::

## 选择合适的路径

| 目标 | 推荐起点 | 主要风险 |
| --- | --- | --- |
| 私密的交互式使用 | 在本地运行现成的量化模型 | 如果配置如此，模型仍可能调用联网工具或记录提示词 |
| 领域适配 | 在精选数据集上做 LoRA 或 QLoRA | 数据泄漏、记忆化，以及评估不充分 |
| 共享 API | 置于认证之后的专用推理服务器 | 成本失控、提示词滥用，以及意外公开暴露 |
| 从零预训练 | 单独立项的研究项目 | 极高的算力、数据治理与评估负担 |

对多数独立研究者而言，本地推理加检索、或一个小规模 LoRA 实验，都比全参数训练更合适。

## 1. 用 Ollama 在本地运行模型

Ollama 在 macOS、Linux 和 Windows 上提供了直观的本地运行时。请使用[官方下载页](https://ollama.com/download)的最新安装程序；在支持 shell 安装脚本的系统上：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

浏览当前的模型库，挑选符合你内存预算的模型，并替换下面的 `<model-name>`：

```bash
ollama pull <model-name>
ollama run <model-name>
ollama list
```

不要只按参数量选模型。请检查它的许可证、上下文长度、支持语言、量化方式、工具调用格式和预期用途。

### 测试本地 API

开发期间请让服务只绑定到本机。一个最简的非流式请求如下：

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "<model-name>",
  "messages": [
    {"role": "user", "content": "Explain canopy photosynthesis in three sentences."}
  ],
  "stream": false
}'
```

如果离线运行是硬性要求，请确认整个流程在离线状态下确实能工作。模型下载到本地，并不证明它周边的每一处集成都是离线的。

## 2. 适配模型之前，先核对硬件

内存需求随模型架构、精度、优化器、序列长度、批大小，以及激活值或优化器状态是否卸载而变化。请避免给出单一的「最低 GPU 要求」结论。

不如生成一份简短的环境报告：

```python
import json
import platform

report = {
    "platform": platform.platform(),
    "python": platform.python_version(),
}

try:
    import torch

    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            1,
        )
except ImportError:
    report["torch"] = None

print(json.dumps(report, indent=2))
```

请使用官网的选择器来安装 PyTorch，以便构建版本与操作系统和加速器匹配。不要把某个固定的 CUDA wheel 命令照抄到 Apple Silicon、纯 CPU，或 CUDA 版本不同的系统上。

## 3. 微调之前先准备数据

微调数据集应当可追溯、其许可允许该用途，并且在开始实验前就完成切分。

1. 定义任务，以及可度量的成功标准。
2. 移除未经授权用于训练的个人信息、机密、受版权保护和重复的内容。
3. 建立训练、验证和测试集，避免近重复样本造成泄漏。
4. 使用所选模型的对话模板来组织样本，而不是自创一套通用提示格式。
5. 保留一份数据集卡片，记录来源、排除项、变换和已知偏差。
6. 在启动长时间训练前，先抽样检查渲染后的提示词与标签。

对科研工作而言，请按真正的泛化单元来切分。例如，来自同一株植物、同一小区、同一站点或同一次采集的样本，如果分散到训练集和测试集中会虚高性能，就不应这样切分。

## 4. 优先选择参数高效适配

LoRA 和 QLoRA 只更新少量适配器参数，同时保持大部分基座权重不变。它们通常能降低内存和存储需求，但并不保证带来更好的事实性或领域表现。

请参照 [Unsloth](https://github.com/unslothai/unsloth) 或 [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) 这类持续维护的训练框架的最新官方说明，并固定一个已验证可用的环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip freeze > requirements-lock.txt
```

你的训练记录应包含：

- 基座模型及其不可变的修订版本；
- 分词器与对话模板；
- LoRA 的目标模块、rank、alpha 和 dropout；
- 序列长度、有效批大小、优化器和学习率；
- 随机种子与数据切分标识；
- 检查点选择规则与评估结果；
- 墙钟时间、加速器型号和峰值内存。

全参数微调和预训练需要单独的算力规划、分布式训练验证，以及强得多的数据治理。它们被有意排除在这份入门工作流之外。

## 5. 评估真正重要的行为

仅有训练损失，不能证明模型有用。请在训练前就构建评估集，并在选择检查点的过程中保持它不变。

| 评估层面 | 示例 |
| --- | --- |
| 任务质量 | 精确匹配、结构化输出有效性、领域评分表、检索忠实度 |
| 稳健性 | 复述改写、缺失上下文、证据冲突、超长输入 |
| 安全性 | 敏感数据召回、提示注入、不安全的工具调用请求 |
| 运行 | 延迟、吞吐、峰值内存、失败率 |
| 人工评审 | 带书面标准的盲测成对偏好比较 |

请报告不确定性和失败案例。BLEU 或困惑度对特定任务可能有参考价值，但两者都不是正确性或有用性的通用度量。

## 6. 先本地部署，再考虑对网络提供服务

对单台工作站而言，运行时自带的本地 API 通常就够了。若需要更高吞吐的 GPU 服务，请查阅最新的 [vLLM 支持模型文档](https://docs.vllm.ai/en/latest/models/supported_models.html)及其部署指南。

在接受远程请求之前，至少要加上：

- 认证与授权；
- 请求体大小、token、并发和速率限制；
- 明确的网络绑定与防火墙规则；
- 超时、取消与背压机制；
- 默认不记录提示词的结构化日志；
- 模型与提示词模板的版本上报；
- 能区分「进程存活」与「模型就绪」的健康检查；
- 滥用测试与回滚方案。

绝不要把未认证的开发服务器或模型面板直接暴露到公网。Gradio 的 `share=True` 隧道同样是一个公开端点，而不是私有的本地界面。

## 排障原则

### 显存不足

先降低序列长度和批大小。然后再考虑梯度累积、梯度检查点、量化、适配器训练，或换用更小的模型。每一处改动都要记录，因为它们会改变质量和速度。

### 软件包或 CUDA 不匹配

新建一个干净环境，分别独立验证驱动、CUDA 运行时、PyTorch 构建和可选的注意力内核。在基础模型能加载之前，不要安装可选的加速库。

### 训练损失不下降

检查渲染后的样本与标签，与一个小规模的过拟合测试作对比，核实学习率，并确认适配器参数确实收到了梯度。步数再多也修不好格式错误的数据。

### 基准分数好，实际用起来差

检查数据泄漏、提示词模板差异、检索质量，以及测试集是否代表真实的部署人群。把观察到的失败案例只加入新的开发集，不要事后补进留出的测试集。

## 可复现性清单

- [ ] 已记录模型许可证与修订版本
- [ ] 已记录数据集来源与切分策略
- [ ] 已锁定依赖
- [ ] 已记录硬件与运行时
- [ ] 提示词与生成参数已纳入版本管理
- [ ] 适配前已评估基线
- [ ] 留出测试集保持未被使用
- [ ] 已测试安全与隐私边界
- [ ] 已报告局限与失败案例

## 延伸阅读

- [Ollama 文档](https://ollama.com/download)
- [vLLM 支持模型](https://docs.vllm.ai/en/latest/models/supported_models.html)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui)
- [Hugging Face 模型库](https://huggingface.co/Qwen/Qwen2-7B)

*工作流核验时间：2026 年 7 月。复现命令前请重新查阅上游文档。*
