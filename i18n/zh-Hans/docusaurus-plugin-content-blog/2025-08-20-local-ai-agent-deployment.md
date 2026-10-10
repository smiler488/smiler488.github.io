---
slug: local-ai-agent-deployment
title: "本地 AI 助手与智能体：安全部署指南"
description: "从本地 Ollama 助手开始，逐步加入检索和受控的工具调用，并处理好隐私、安全和版本管理。"
authors: [liangchao]
tags: [artificial-intelligence, local-ai, reproducible-research, python]
image: /img/blog-default.jpg
category: 人工智能与机器学习
article_type: 技术指南
---

在本地运行模型，可以减少对云端推理接口的依赖，让提示词留在自己的硬件上。但本地运行模型并不等于有了**智能体**；如果外围应用用到了远程搜索、遥测、云端向量化、公网隧道或联网工具，也谈不上隐私。

本文从一个本地助手开始，逐步加入检索和工具调用，每次只放开一项能力。每项新能力都应当可观察、可撤销，权限不超过任务所需。

<!-- truncate -->

## 助手、RAG 还是智能体？

| 类型 | 增加了什么 | 主要风险 |
| --- | --- | --- |
| 本地助手 | 根据提示词生成回答的模型 | 编造内容、服务被意外暴露到网络 |
| 检索增强生成（RAG） | 在指定文档库中检索 | 数据泄露、索引过时、回答缺乏依据 |
| 调用工具的助手 | 以结构化方式调用指定的函数 | 参数错误、产生意外后果 |
| 智能体 | 根据状态自主选择并连续执行多个操作 | 错误层层累积、自主权过大、责任不清 |

和 Ollama 对话只是本地助手。只有加入了明确的执行循环、工具接口约定、状态管理、停止条件和权限控制之后，才能称为智能体。

## 1. 先划定使用边界

安装之前，先把下面这些问题想清楚并记下来：

- 哪些密级的数据可以写进提示词？
- 是否必须在断网环境下工作？
- 哪些用户和设备可以访问？
- 哪些工具只读，哪些能修改文件或外部系统？
- 哪些操作需要人工确认？
- 记录哪些日志、保存多久、谁能查看？
- 模型、提示词、索引和工具的版本怎样标识？
- 出问题时如何回滚和处置？

“本地模型”只说明推理在哪里运行，回答不了上面这些系统层面的问题。

## 2. 在本机启动 Ollama

从[官方下载页](https://ollama.com/download)安装 Ollama；支持脚本安装的系统也可以执行：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

在官方模型库中选一个内存放得下、许可证合适的模型，替换 `<model-name>`：

```bash
ollama pull <model-name>
ollama run <model-name>
ollama list
```

测试本地对话接口：

```python
import requests

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "<model-name>",
        "messages": [
            {
                "role": "user",
                "content": "Summarize this experiment plan in five bullets.",
            }
        ],
        "stream": False,
    },
    timeout=120,
)
response.raise_for_status()
print(response.json()["message"]["content"])
```

本例只需额外安装一个依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install requests
python -m pip freeze > requirements-lock.txt
```

测试阶段让服务只监听本机。在认定它“只有自己能访问”之前，用系统自带的网络工具确认实际监听的地址。

## 3. 记录运行环境

下面这段 Python 代码在主流桌面系统上都能运行，不依赖 `free`、`nproc` 等 Linux 专用命令：

```python
import json
import os
import platform
import shutil

report = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "cpu_count": os.cpu_count(),
    "ollama_path": shutil.which("ollama"),
    "docker_path": shutil.which("docker"),
}

try:
    import torch

    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
except ImportError:
    report["torch"] = None

print(json.dumps(report, indent=2))
```

内存需求取决于模型、量化方式、上下文长度、并发请求数和运行开销。请在实际机器上对所选模型实测，不要笼统地给所有 7B 或 13B 模型定一个硬件门槛。

## 4. 加入检索，同时保留出处

一个便于维护的本地 RAG 流程分五步：

1. **导入：** 只解析白名单内的文件；不支持或加密的文件直接报错，不悄悄跳过。
2. **切分：** 保留文档结构和固定的编号。
3. **向量化：** 记录所用嵌入模型的名称和版本。
4. **检索：** 每个候选片段附带来源、页码或章节、相似度分数和索引版本。
5. **生成：** 提示词要求模型依据检索到的内容回答，并注明出处。

每个片段都保留原始文档的编号。检索和回答要分开评估：如果相关段落根本没检索到，改提示词也补不上缺失的证据。

### 最小评估集

准备一组问题，并标注每题应当命中的原文段落，其中包括：

- 能在文档中找到答案的问题；
- 文档中没有答案的问题；
- 内容相互矛盾的文档；
- 已被新版本取代的旧文档；
- 表格、图题和长章节；
- 文档中夹带的恶意指令。

统计检索召回率、引用准确率、无依据回答的比例、响应时间，以及缺少证据时模型的表现。

## 5. 用窄接口提供工具

第一个交给模型的工具，不应是通用 shell、不受限制的文件系统访问或权限宽泛的 API 令牌。每项能力都应包装成一个小而明确、带类型约束的函数：

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReadTextRequest:
    relative_path: str


def read_project_text(request: ReadTextRequest, workspace: Path) -> str:
    root = workspace.resolve()
    target = (root / request.relative_path).resolve()

    if root not in target.parents:
        raise ValueError("Path escapes the approved workspace")
    if target.suffix.lower() not in {".md", ".txt", ".csv"}:
        raise ValueError("File type is not allowed")
    if target.stat().st_size > 1_000_000:
        raise ValueError("File exceeds the read limit")

    return target.read_text(encoding="utf-8")
```

这个例子只读，且限定在工作目录内。真正上线的工具还需要结构化的错误处理、审计编号、拒绝情形的测试，以及对符号链接和并发竞争等边界情况的防护。

每个工具都要明确：

- 输入格式和大小限制；
- 调用身份和最小权限凭据；
- 允许访问的资源和禁止访问的路径；
- 超时、重试和重复执行时的行为；
- 重要操作执行前的预览；
- 需要用户确认的情形；
- 一条结构化、已脱敏的审计记录；
- 明确的停止条件。

检索到的文本、网页、邮件和文档都是不可信的数据，其中可能夹带试图操纵模型的指令。

## 6. 约束智能体的执行循环

安全的执行循环从设计上就有边界：

```text
接收请求
  → 判断数据密级和权限
  → 提出计划
  → 从白名单中选择一个工具
  → 校验参数
  → 需要时请求用户确认
  → 带超时执行
  → 记录脱敏后的结果
  → 停止，或在严格的步数预算内继续
```

对步数、运行时间、token 数、工具调用次数、文件数量和重试次数都设上限。模型不能自行提高这些上限，也不能给自己添加新工具。

## 7. 共享之前先做安全加固

只要服务对网络开放，至少要做到：

- 只监听预期的网络接口；
- 认证用户身份，并对每个工具单独授权；
- 密钥在代码之外生成，缺失时拒绝启动；
- 在不可信网络上使用 TLS；
- 限制请求大小、上下文长度、并发数和请求频率；
- 文档库和向量数据库不对公网开放端口；
- 日志中对提示词、文档、凭据和个人信息脱敏；
- 模型运行环境与高权限工具相互隔离；
- 扫描依赖漏洞，容器镜像按版本号或摘要固定；
- 备份索引和配置，并实际演练恢复；
- 提供紧急停止开关，出事后立即吊销凭据。

用字符黑名单过滤并不能防御提示注入。同样，使用 `your-secret-key` 这类默认 JWT 密钥，会让一次配置失误直接变成认证被绕过。

## Docker 与 GPU

Compose 文件里预留了 GPU，并不会让基于 CPU 的 Python 基础镜像自动支持 CUDA。请使用官方说明适配目标运行时和驱动的推理镜像，或者把 Ollama 放在应用容器之外运行，再通过受限的本地端口调用。

不要暴露没有认证的向量数据库、模型接口或开发界面。需要可复现的部署，不要使用 `latest` 镜像标签。请按最新的 [Docker 安装指南](https://get.docker.com)和 NVIDIA 容器文档配置，不要照搬旧教程里的 `apt-key` 或 `nvidia-docker2` 脚本。

## 上线前的测试

在处理真实科研数据之前，测试以下情况：

- 请求处理过程中服务重启；
- 格式错误或超大的输入；
- 检索文档中夹带的提示注入；
- 试图越出白名单的工具参数；
- 密钥缺失或凭据过期；
- 模型或向量库无法连接；
- 接近资源上限时的并发请求；
- 取消请求和超时；
- 从备份恢复；
- 模型或索引回滚。

失败的测试结果也要记录，而不只是成功的演示。

## 可复现记录

```json
{
  "runtime": "Ollama",
  "runtime_version": "<version>",
  "model": "<model-name>",
  "model_digest": "<digest>",
  "prompt_version": "assistant-v1",
  "embedding_model": "<model-and-revision>",
  "index_version": "documents-2026-07-16",
  "tool_policy_version": "read-only-v1",
  "network_mode": "localhost-only",
  "evaluation_set": "local-agent-eval-v1"
}
```

这份记录与评估结果、部署配置一起保存。

## 实验室中的相关工具

[多模态 AI 求解器](/app/solver)是一个浏览器客户端，使用用户自己的 AI 服务商密钥发送多模态请求。它的[教程](/docs/tutorial-apps/ai-solver-tutorial)说明了服务商、模型、浏览器权限、屏幕截取和隐私方面的注意事项。它调用的是云端模型，不是本地离线运行。

## 最终检查

- [ ] 本地助手已能正常使用，再加入检索或工具
- [ ] 已实际观察并记录网络连接情况
- [ ] 已固定模型和依赖版本
- [ ] 已评估检索结果的引用准确性
- [ ] 工具接口窄、权限最小
- [ ] 重要操作需要人工确认
- [ ] 执行循环有硬性上限和停止条件
- [ ] 密钥缺失时拒绝启动，且从不使用默认值
- [ ] 日志已脱敏，访问受控
- [ ] 回滚和紧急停止流程已演练
