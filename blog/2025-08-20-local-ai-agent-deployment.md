---
slug: local-ai-agent-deployment
title: "Local AI Assistants and Agents: A Secure, Reproducible Deployment Guide"
description: "A practical path from a local Ollama assistant to retrieval and controlled tool use, with explicit privacy, security, versioning, and deployment boundaries."
authors: [liangchao]
tags: [artificial-intelligence, local-ai, reproducible-research, python]
image: /img/blog-default.jpg
category: "AI & machine learning"
article_type: Technical guide
---

Running a model locally can reduce dependence on a hosted inference API and keep prompts on controlled hardware. It does not automatically create an **agent**, and it does not guarantee privacy if the surrounding application uses remote search, telemetry, hosted embeddings, public tunnels, or networked tools.

This guide begins with a local assistant, then adds retrieval and optional tool use one boundary at a time. Each new capability should be observable, reversible, and no more privileged than the task requires.

<!-- truncate -->

## Assistant, RAG system, or agent?

| System | What it adds | Main risk |
| --- | --- | --- |
| Local assistant | A model that generates responses from prompts | Hallucination and accidental network exposure |
| Retrieval-augmented generation (RAG) | Search over a controlled document collection | Data leakage, stale indexes, and unsupported answers |
| Tool-using assistant | Structured calls to approved functions | Incorrect arguments and unintended side effects |
| Agent | A loop that chooses and executes multiple actions using state | Compounding errors, excessive autonomy, and unclear accountability |

An interactive chat with Ollama is a local assistant. Call it an agent only after adding an explicit action loop, tool contracts, state, stopping conditions, and authorization controls.

## 1. Write the deployment boundary first

Document these decisions before installation:

- Which data classifications may enter prompts?
- Must the system work with the network disabled?
- Which users and devices may access it?
- Which tools are read-only, and which can change files or external systems?
- Which actions require confirmation?
- What is logged, for how long, and who can read it?
- How will model, prompt, index, and tool versions be identified?
- What is the rollback and incident-response path?

“Local model” describes where inference runs. It does not answer these system-level questions.

## 2. Start with Ollama on localhost

Install Ollama from the [official download page](https://ollama.com/download). Where the shell installer is supported:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Select a current model from the official library that fits the available memory and license requirements, then replace `<model-name>`:

```bash
ollama pull <model-name>
ollama run <model-name>
ollama list
```

Test the local chat API:

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

Install the only additional dependency used by this example:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install requests
python -m pip freeze > requirements-lock.txt
```

Keep the service on localhost while testing. Confirm listening interfaces with operating-system network tools before assuming it is private.

## 3. Record the environment portably

This Python report works across major desktop platforms without Linux-only commands such as `free` or `nproc`:

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

Memory requirements depend on model, quantization, context, parallel requests, and runtime overhead. Benchmark the chosen artifact on the actual machine rather than labeling all 7B or 13B models with one hardware threshold.

## 4. Add retrieval without hiding provenance

A maintainable local RAG pipeline has five explicit stages:

1. **Ingest:** allowlisted files are parsed; unsupported or encrypted files fail visibly.
2. **Chunk:** document structure and stable identifiers are preserved.
3. **Embed:** the exact embedding model and revision are recorded.
4. **Retrieve:** candidate chunks include source, page or section, score, and index version.
5. **Generate:** the prompt instructs the model to answer from evidence and cite retrieved sources.

Keep the original document identifier in every chunk. Evaluate retrieval separately from answer generation: if the relevant passage is not retrieved, changing the final prompt cannot repair the evidence gap.

### Minimal evaluation set

Create questions with expected source passages and include:

- answerable questions;
- questions whose answer is absent;
- conflicting documents;
- superseded versions;
- tables, captions, and long sections;
- adversarial text embedded inside a document.

Measure retrieval recall, citation correctness, unsupported-claim rate, latency, and behavior when evidence is missing.

## 5. Add tools through narrow contracts

Do not give a model a general shell, unrestricted filesystem, or broad API token as the first tool. Wrap each capability in a small typed function.

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

This example is read-only and workspace-scoped. A production tool still needs structured error handling, audit identifiers, denial tests, and protection against symbolic-link and race-condition edge cases.

For every tool, define:

- an input schema and size limits;
- authentication context and least-privilege credentials;
- allowed resources and denied paths;
- timeout, retry, and idempotency behavior;
- a preview for consequential actions;
- user confirmation rules;
- a structured, redacted audit event;
- a deterministic stop condition.

Treat retrieved text, webpages, emails, and documents as untrusted data. They can contain instructions designed to manipulate the model.

## 6. Control the agent loop

A safe loop should be bounded by design:

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

Set limits for steps, wall time, tokens, tool calls, file volume, and retries. The model must not be able to increase its own limits or grant itself new tools.

## 7. Secure the service before sharing it

Minimum controls for any networked deployment include:

- bind only to the intended interface;
- authenticate users and authorize each tool separately;
- generate secrets outside source code and fail closed when missing;
- use TLS across untrusted networks;
- set request, context, concurrency, and rate limits;
- keep document stores and vector databases off public ports;
- redact prompts, documents, credentials, and personal data from logs;
- separate model execution from privileged tools;
- scan dependencies and pin container images by version or digest;
- back up indexes and configuration with tested restore procedures;
- provide a kill switch and revoke credentials after an incident.

Character blacklists are not a defense against prompt injection. Likewise, a default JWT secret such as `your-secret-key` turns a configuration error into an authentication bypass.

## Docker and GPU notes

A CPU Python base image does not gain CUDA support merely because a Compose file reserves a GPU. Use an inference image documented for the target runtime and driver, or keep Ollama outside the application container and call its restricted local endpoint.

Do not expose an unauthenticated vector database, model API, or development UI. Avoid `latest` image tags in a reproducible deployment. Follow current [Docker installation guidance](https://get.docker.com) and NVIDIA container documentation rather than copying an old `apt-key` or `nvidia-docker2` setup script.

## Operational tests

Before allowing real research data, test:

- service restart during a request;
- malformed and oversized inputs;
- prompt injection in retrieved documents;
- tool arguments that escape the allowlist;
- missing secrets and expired credentials;
- unreachable model or vector store;
- concurrent requests near the resource limit;
- cancellation and timeout behavior;
- restoration from backup;
- model or index rollback.

Record failure outcomes, not only successful demonstrations.

## Reproducibility manifest

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

Store this record with evaluation results and deployment configuration.

## Related App Lab tool

[Multimodal AI Solver](/app/solver) is a browser client for user-supplied AI-provider credentials and multimodal requests. Its [tutorial](/docs/tutorial-apps/ai-solver-tutorial) describes provider, model, browser-permission, screen-capture, and privacy boundaries. It is not a local offline model runtime.

## Final checklist

- [ ] Local assistant works before retrieval or tools are added
- [ ] Network behavior has been observed and documented
- [ ] Model and dependency versions are pinned
- [ ] Retrieval citations are evaluated
- [ ] Tools are narrow and least-privileged
- [ ] Consequential actions require confirmation
- [ ] Agent loop has hard budgets and a stop condition
- [ ] Secrets fail closed and never use a default value
- [ ] Logs are redacted and access-controlled
- [ ] Rollback and kill-switch procedures are tested

*Workflow reviewed: July 2026. Re-check Ollama, container-runtime, and framework documentation before deployment.*
