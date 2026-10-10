---
slug: local-llm-training-guide-en
title: "本地大模型：运行、微调与部署"
description: "如何选择本地模型并用 Ollama 运行，规划参数高效微调，评估效果，以及安全地对外提供服务。"
authors: [liangchao]
tags: [artificial-intelligence, machine-learning, local-ai, reproducible-research]
image: /img/blog-default.jpg
category: "人工智能与机器学习"
article_type: 技术指南
---

当数据必须留在自有硬件上、需要离线运行，或实验需要固定的模型与软件环境时，本地大语言模型是合适的选择。至于是否更省钱、更快或更私密，则取决于模型规模、硬件、网络设置以及服务的对外开放方式。

本文把常被混为一谈的三件事分开讲：**运行模型**、**微调模型**和**部署模型**。能用最简单的方式回答研究问题，就不必做更复杂的事。

<!-- truncate -->

:::note 版本

模型名称、软件包 API、CUDA 版本和硬件要求更新很快。每次实验都应记录模型版本、依赖锁定文件、操作系统、驱动、加速器、提示词模板和测试日期。

:::

## 选择合适的路径

| 目标 | 推荐起点 | 主要风险 |
| --- | --- | --- |
| 私密的交互式使用 | 在本地运行现成的量化模型 | 若配置了联网工具或日志，提示词仍可能外传或被记录 |
| 领域适配 | 在整理好的数据集上做 LoRA 或 QLoRA | 数据泄漏、模型死记训练数据、评估不充分 |
| 共享 API | 带身份认证的专用推理服务器 | 成本失控、提示词滥用、服务被意外公开 |
| 从零预训练 | 单独立项的研究项目 | 算力、数据管理和评估的成本都极高 |

对多数独立研究者而言，本地推理加检索、或一个小规模 LoRA 实验，都比全参数训练更合适。

## 1. 用 Ollama 在本地运行模型

Ollama 是一个简单易用的本地运行工具，支持 macOS、Linux 和 Windows。请从[官方下载页](https://ollama.com/download)安装最新版本；支持脚本安装的系统也可以执行：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

在模型库中选一个内存放得下的模型，替换下面的 `<model-name>`：

```bash
ollama pull <model-name>
ollama run <model-name>
ollama list
```

选模型不能只看参数量，还要看许可证、上下文长度、支持的语言、量化方式、工具调用格式和适用场景。

### 测试本地 API

开发阶段让服务只监听本机。最简单的非流式请求如下：

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "<model-name>",
  "messages": [
    {"role": "user", "content": "Explain canopy photosynthesis in three sentences."}
  ],
  "stream": false
}'
```

如果必须离线运行，请断网实际测试整个流程。模型下载到了本地，并不代表与之相连的其他组件都不联网。

## 2. 微调之前先确认硬件

显存需求取决于模型结构、精度、优化器、序列长度、批大小，以及激活值和优化器状态是否卸载到内存，很难用一个“最低显卡要求”概括。

更实用的做法是先生成一份环境报告：

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

PyTorch 请按官网生成的命令安装，确保与操作系统和加速硬件匹配。不要把某条针对特定 CUDA 版本的安装命令照搬到 Apple 芯片、纯 CPU 或 CUDA 版本不同的机器上。

## 3. 微调之前先准备数据

微调数据要来源可查、许可允许用于训练，并在实验开始前就划分好。

1. 明确任务和可量化的成功标准。
2. 去除个人信息、涉密内容、未获授权的版权内容和重复样本。
3. 划分训练集、验证集和测试集，防止相似样本跨集合泄漏。
4. 按所选模型自带的对话模板组织样本，不要自己发明格式。
5. 写一份数据集说明，记录来源、剔除内容、处理步骤和已知偏差。
6. 开始长时间训练前，抽查套用模板后的提示词和标签。

科研数据要按真正需要推广的单元划分。例如来自同一植株、同一小区、同一站点或同一次采集的样本，如果分散在训练集和测试集中，会让结果虚高。

## 4. 优先采用参数高效微调

LoRA 和 QLoRA 只训练少量适配器参数，基座模型的权重基本不动，因此显存和存储开销小得多；但它们不一定能让模型更准确或更懂专业领域。

请按 [Unsloth](https://github.com/unslothai/unsloth)、[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) 等持续维护的训练框架的最新官方说明操作，并锁定一套验证可用的环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip freeze > requirements-lock.txt
```

训练记录应包括：

- 基座模型及其确切版本；
- 分词器与对话模板；
- LoRA 的目标模块、rank、alpha 和 dropout；
- 序列长度、有效批大小、优化器和学习率；
- 随机种子与数据切分标识；
- 检查点选择规则与评估结果；
- 实际耗时、加速卡型号和峰值显存。

全参数微调和预训练需要专门的算力规划、分布式训练调试和严格得多的数据管理，不在本文讨论范围内。

## 5. 评估真正重要的能力

训练损失下降不能说明模型有用。评估集要在训练前建好，挑选检查点时保持不变。

| 评估层面 | 示例 |
| --- | --- |
| 任务质量 | 精确匹配、结构化输出有效性、领域评分表、检索忠实度 |
| 稳健性 | 换种说法提问、缺少上下文、信息相互矛盾、超长输入 |
| 安全性 | 是否泄露敏感数据、提示注入、危险的工具调用 |
| 运行 | 延迟、吞吐、峰值内存、失败率 |
| 人工评审 | 按书面标准做盲评的两两比较 |

结果要报告不确定性和失败案例。BLEU 和困惑度对某些任务有参考价值，但都不能普遍衡量回答是否正确、是否有用。

## 6. 先在本机使用，再考虑对外服务

单台工作站上，运行工具自带的本地 API 通常就够用了。需要更高吞吐的 GPU 服务时，可参考最新的 [vLLM 支持模型文档](https://docs.vllm.ai/en/latest/models/supported_models.html)及其部署指南。

开放远程访问之前，至少要做到：

- 认证与授权；
- 限制请求大小、token 数、并发数和请求频率；
- 明确监听地址，并配置防火墙规则；
- 支持超时、取消和过载保护；
- 结构化日志，默认不记录提示词内容；
- 能查询当前模型和提示词模板的版本；
- 健康检查能区分“进程在运行”和“模型已就绪”；
- 做过滥用测试，并有回滚方案。

不要把没有认证的开发服务器或模型管理界面直接开放到公网。Gradio 的 `share=True` 生成的也是公开链接，并不是只有自己能访问的本地界面。

## 常见问题

### 显存不足

先减小序列长度和批大小，再考虑梯度累积、梯度检查点、量化、适配器训练或换用更小的模型。每项改动都会影响效果和速度，需要记录下来。

### 软件包或 CUDA 不匹配

新建一个干净的环境，逐一确认驱动、CUDA 运行时、PyTorch 版本和可选的注意力加速组件。基础模型能正常加载之前，先不要装可选的加速库。

### 训练损失不下降

检查套用模板后的样本和标签，先用少量数据做过拟合测试，核对学习率，并确认适配器参数确实在更新。数据格式有错，训练再久也没用。

### 测试分数高，实际效果差

检查是否存在数据泄漏、提示词模板是否一致、检索质量如何，以及测试集能否代表真实使用场景。发现的失败案例只加入新的开发集，不要事后补进测试集。

## 可复现性清单

- [ ] 已记录模型许可证与修订版本
- [ ] 已记录数据来源和划分方式
- [ ] 已锁定依赖
- [ ] 已记录硬件与运行时
- [ ] 提示词与生成参数已纳入版本管理
- [ ] 微调前已评估基线模型
- [ ] 测试集未在开发过程中使用
- [ ] 已测试安全和隐私防护
- [ ] 已报告局限与失败案例

## 延伸阅读

- [Ollama 文档](https://ollama.com/download)
- [vLLM 支持模型](https://docs.vllm.ai/en/latest/models/supported_models.html)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui)
- [Hugging Face 模型库](https://huggingface.co/Qwen/Qwen2-7B)
