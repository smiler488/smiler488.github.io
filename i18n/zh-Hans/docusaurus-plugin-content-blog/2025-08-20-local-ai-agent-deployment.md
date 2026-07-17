---
slug: local-ai-agent-deployment
title: "本地 AI 助手与智能体：一份安全、可复现的部署指南"
description: "一条从本地 Ollama 助手到检索和受控工具使用的实用路径，并明确隐私、安全、版本和部署边界。"
authors: [liangchao]
tags: [artificial-intelligence, local-ai, reproducible-research, python]
image: /img/blog-default.jpg
category: 人工智能与机器学习
article_type: 技术指南
---

在本地运行模型可以减少对托管推理 API 的依赖，并把提示保留在受控硬件上。它并不自动创建一个**智能体**，如果周边应用使用了远程搜索、遥测、托管嵌入、公共隧道或联网工具，它也不保证隐私。

本指南从一个本地助手开始，然后每次一个边界地添加检索和可选的工具使用。每个新能力都应是可观测、可逆的，且权限不超过任务所需。

<!-- truncate -->

## 助手、RAG 系统还是智能体？

| 系统 | 它增加什么 | 主要风险 |
| --- | --- | --- |
| 本地助手 | 一个从提示生成回复的模型 | 幻觉和意外的网络暴露 |
| 检索增强生成（RAG） | 在受控文档集合上的搜索 | 数据泄漏、陈旧索引和无据回答 |
| 使用工具的助手 | 对已批准函数的结构化调用 | 错误参数和非预期副作用 |
| 智能体 | 一个使用状态选择并执行多个动作的循环 | 错误累积、过度自主和责任不清 |

与 Ollama 的交互式聊天是一个本地助手。只有在添加了明确的动作循环、工具契约、状态、停止条件和授权控制之后，才把它称作智能体。

## 1. 先写好部署边界

安装之前记录这些决定：

- 哪些数据分级可以进入提示？
- 系统是否必须在断网下工作？
- 哪些用户和设备可以访问它？
- 哪些工具是只读的，哪些可以改变文件或外部系统？
- 哪些动作需要确认？
- 记录什么、记录多久、谁可以读取？
- 模型、提示、索引和工具版本如何标识？
- 回滚和事件响应路径是什么？

“本地模型”描述的是推理在哪里运行。它并不回答这些系统层面的问题。

## 2. 在 localhost 上启动 Ollama

从[官方下载页](https://ollama.com/download)安装 Ollama。在支持 shell 安装脚本的地方：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

从官方库中选择一个适合可用内存和许可要求的当前模型，然后替换 `<model-name>`：

```bash
ollama pull <model-name>
ollama run <model-name>
ollama list
```

测试本地聊天 API：

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

安装本示例使用的唯一额外依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install requests
python -m pip freeze > requirements-lock.txt
```

测试期间把服务保留在 localhost 上。在假定它私有之前，用操作系统的网络工具确认监听接口。

## 3. 可移植地记录环境

这份 Python 报告在主流桌面平台上都能运行，不依赖 `free` 或 `nproc` 等 Linux 专属命令：

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

内存需求取决于模型、量化、上下文、并行请求和运行时开销。在实际机器上对所选产物进行基准测试，而不是用一个硬件阈值给所有 7B 或 13B 模型贴标签。

## 4. 添加检索但不隐藏来源

一个可维护的本地 RAG 流程有五个明确阶段：

1. **摄取：** 解析允许清单中的文件；不支持或加密的文件会显式失败。
2. **分块：** 保留文档结构和稳定标识符。
3. **嵌入：** 记录确切的嵌入模型和版本。
4. **检索：** 候选分块包含来源、页码或章节、分数和索引版本。
5. **生成：** 提示指示模型从证据回答并引用检索到的来源。

在每个分块中保留原始文档标识符。把检索与答案生成分开评估：如果相关段落没有被检索到，改变最终提示无法弥补证据缺口。

### 最小评估集

创建带有预期来源段落的问题，并包含：

- 可回答的问题；
- 答案缺失的问题；
- 相互冲突的文档；
- 被取代的版本；
- 表格、题注和长章节；
- 嵌在文档内部的对抗性文本。

测量检索召回率、引用正确率、无据声明率、延迟，以及证据缺失时的行为。

## 5. 通过窄契约添加工具

不要把一个通用 shell、不受限的文件系统或宽泛的 API 令牌作为第一个工具交给模型。把每个能力包装在一个小型带类型函数中。

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

此示例是只读且限定在工作区内的。生产工具仍需结构化错误处理、审计标识符、拒绝测试，以及对符号链接和竞态条件边缘情况的防护。

对每个工具，定义：

- 输入模式和大小限制；
- 认证上下文和最小权限凭据；
- 允许的资源和被拒绝的路径；
- 超时、重试和幂等行为；
- 对重要动作的预览；
- 用户确认规则；
- 一个结构化、脱敏的审计事件；
- 一个确定性的停止条件。

把检索到的文本、网页、邮件和文档视为不受信任的数据。它们可能包含旨在操纵模型的指令。

## 6. 控制智能体循环

一个安全的循环应在设计上有界：

```text
receive request
  → classify data and permissions
  → propose a plan
  → choose one allowlisted tool
  → validate arguments
  → request confirmation when required
  → execute with timeout
  → record a redacted result
  → stop, or continue within a strict step budget
```

为步数、挂钟时间、token、工具调用、文件量和重试设置限制。模型必须不能提高自己的限制或给自己授予新工具。

## 7. 共享前先加固服务

任何联网部署的最低控制包括：

- 仅绑定到预期接口；
- 认证用户并分别授权每个工具；
- 在源代码之外生成密钥，缺失时失败关闭；
- 在不受信任网络上使用 TLS；
- 设置请求、上下文、并发和速率限制；
- 让文档库和向量数据库远离公共端口；
- 从日志中脱敏提示、文档、凭据和个人数据；
- 把模型执行与特权工具分离；
- 扫描依赖并按版本或摘要固定容器镜像；
- 备份索引和配置并测试恢复流程；
- 提供一个紧急停止开关，并在事件后撤销凭据。

字符黑名单不是对抗提示注入的防御手段。同样，诸如 `your-secret-key` 这样的默认 JWT 密钥会把一个配置错误变成认证绕过。

## Docker 与 GPU 说明

一个 CPU 的 Python 基础镜像不会仅因为 Compose 文件预留了 GPU 就获得 CUDA 支持。使用为目标运行时和驱动有文档说明的推理镜像，或者把 Ollama 放在应用容器之外并调用其受限的本地端点。

不要暴露未认证的向量数据库、模型 API 或开发 UI。在可复现的部署中避免 `latest` 镜像标签。遵循当前的 [Docker 安装指南](https://get.docker.com) 和 NVIDIA 容器文档，而不是照抄旧的 `apt-key` 或 `nvidia-docker2` 设置脚本。

## 运维测试

在允许真实科研数据之前，测试：

- 请求期间的服务重启；
- 畸形和超大输入；
- 检索文档中的提示注入；
- 逃出允许清单的工具参数；
- 缺失的密钥和过期的凭据；
- 不可达的模型或向量库；
- 接近资源限制时的并发请求；
- 取消和超时行为；
- 从备份恢复；
- 模型或索引回滚。

记录失败结果，而不仅仅是成功演示。

## 可复现性清单

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

把此记录与评估结果和部署配置一起保存。

## 相关 App Lab 工具

[多模态 AI 求解器](/app/solver) 是一个浏览器客户端，使用用户自备的 AI 提供商凭据进行多模态请求。其[教程](/docs/tutorial-apps/ai-solver-tutorial)说明了提供商、模型、浏览器权限、屏幕捕获和隐私边界。它不是本地离线模型运行时。

## 最终清单

- [ ] 在添加检索或工具之前本地助手已可用
- [ ] 已观测并记录网络行为
- [ ] 模型和依赖版本已固定
- [ ] 检索引用已评估
- [ ] 工具窄且最小权限
- [ ] 重要动作需要确认
- [ ] 智能体循环有硬预算和停止条件
- [ ] 密钥失败关闭且绝不使用默认值
- [ ] 日志已脱敏且访问受控
- [ ] 回滚和紧急停止流程已测试

*工作流审阅：2026 年 7 月。部署前请重新查阅 Ollama、容器运行时和框架文档。*
