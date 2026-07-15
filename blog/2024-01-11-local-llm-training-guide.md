---
slug: local-llm-training-guide-en
title: "Local LLMs: A Reproducible Guide to Running, Fine-Tuning, and Serving Models"
description: "A maintainable workflow for choosing a local model, testing it with Ollama, planning parameter-efficient fine-tuning, evaluating results, and exposing a service safely."
authors: [liangchao]
tags: [artificial-intelligence, machine-learning, local-ai, reproducible-research]
image: /img/blog-default.jpg
category: "AI & machine learning"
article_type: Technical guide
---

Local large language models are useful when data must stay on controlled hardware, when offline operation matters, or when an experiment needs a fixed model and software stack. They are not automatically cheaper, faster, or more private: those outcomes depend on model size, hardware, network settings, and how the service is exposed.

This guide separates three jobs that are often mixed together: **running a model**, **adapting a model**, and **serving a model**. Start with the smallest job that answers your research question.

<!-- truncate -->

:::caution Version-sensitive workflow

Model names, package APIs, CUDA builds, and hardware requirements change quickly. Record the model revision, package lockfile, operating system, driver, accelerator, prompt template, and test date for every reproducible experiment. Commands below are starting points, not universal production recipes.

:::

## Choose the right path

| Goal | Recommended starting point | Main risk |
| --- | --- | --- |
| Private interactive use | Run an existing quantized model locally | A model may still call networked tools or log prompts if configured to do so |
| Domain adaptation | LoRA or QLoRA on a curated dataset | Data leakage, memorization, and weak evaluation |
| Shared API | A dedicated inference server behind authentication | Unbounded cost, prompt abuse, and accidental public exposure |
| Pre-training from scratch | A separately designed research project | Very high compute, data governance, and evaluation burden |

For most individual researchers, local inference plus retrieval or a small LoRA experiment is more appropriate than full-parameter training.

## 1. Run a model locally with Ollama

Ollama provides a straightforward local runtime on macOS, Linux, and Windows. Use the current installer from the [official download page](https://ollama.com/download); on systems where the shell installer is supported:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Browse the current model library, select a model that fits your memory budget, and replace `<model-name>` below:

```bash
ollama pull <model-name>
ollama run <model-name>
ollama list
```

Do not select a model only by parameter count. Check its license, context length, supported languages, quantization, tool-calling format, and intended use.

### Test the local API

Keep the service bound to the local machine during development. A minimal non-streaming request is:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "<model-name>",
  "messages": [
    {"role": "user", "content": "Explain canopy photosynthesis in three sentences."}
  ],
  "stream": false
}'
```

Confirm that the process works offline if offline operation is a requirement. A downloaded model does not prove that every surrounding integration is offline.

## 2. Check hardware before adapting a model

Memory demand varies with model architecture, precision, optimizer, sequence length, batch size, and whether activations or optimizer states are offloaded. Avoid a single “minimum GPU” claim.

Use a small environment report instead:

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

Install PyTorch using the selector on the official site so the build matches the operating system and accelerator. Do not copy a fixed CUDA wheel command onto Apple Silicon, CPU-only, or differently versioned CUDA systems.

## 3. Prepare data before fine-tuning

A fine-tuning dataset should be traceable, licensed for the intended use, and split before experimentation.

1. Define the task and a measurable success criterion.
2. Remove personal, confidential, copyrighted, and duplicated material that is not authorized for training.
3. Create train, validation, and test splits that prevent near-duplicate leakage.
4. Format examples using the selected model's chat template rather than inventing a generic prompt format.
5. Keep a dataset card with provenance, exclusions, transformations, and known biases.
6. Inspect a sample of rendered prompts and labels before starting a long run.

For scientific work, split by the true unit of generalization. For example, specimens from the same plant, plot, site, or acquisition session should not be scattered across train and test sets when that would inflate performance.

## 4. Prefer parameter-efficient adaptation

LoRA and QLoRA update a small set of adapter parameters while keeping most base-model weights fixed. They generally reduce memory and storage requirements, but they do not guarantee better factuality or domain performance.

Use the current official instructions for a maintained training framework such as [Unsloth](https://github.com/unslothai/unsloth) or [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl), and pin a known-working environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip freeze > requirements-lock.txt
```

Your training record should include:

- base model and immutable revision;
- tokenizer and chat template;
- LoRA target modules, rank, alpha, and dropout;
- sequence length, effective batch size, optimizer, and learning rate;
- random seed and data split identifiers;
- checkpoint selection rule and evaluation results;
- wall time, accelerator type, and peak memory.

Full-parameter fine-tuning and pre-training require a separate capacity plan, distributed-training validation, and substantially stronger data governance. They are intentionally outside this introductory workflow.

## 5. Evaluate the behavior that matters

Training loss alone is not evidence of a useful model. Build an evaluation set before training and keep it unchanged while selecting checkpoints.

| Evaluation layer | Examples |
| --- | --- |
| Task quality | Exact match, structured-output validity, domain rubric, retrieval faithfulness |
| Robustness | Paraphrases, missing context, conflicting evidence, long inputs |
| Safety | Sensitive-data recall, prompt injection, unsafe tool requests |
| Operations | Latency, throughput, peak memory, failure rate |
| Human review | Blinded pairwise preference with written criteria |

Report uncertainty and failure cases. BLEU or perplexity can be informative for specific tasks, but neither is a general measure of correctness or helpfulness.

## 6. Serve locally before serving a network

For a single workstation, the runtime's local API is often enough. For higher-throughput GPU serving, consult the current [vLLM supported-model documentation](https://docs.vllm.ai/en/latest/models/supported_models.html) and its deployment guide.

Before accepting remote requests, add at least:

- authentication and authorization;
- request-size, token, concurrency, and rate limits;
- explicit network binding and firewall rules;
- timeouts, cancellation, and back-pressure;
- structured logs that exclude prompts by default;
- model and prompt-template version reporting;
- health checks that distinguish “process alive” from “model ready”;
- abuse testing and a rollback plan.

Never expose an unauthenticated development server or a model dashboard directly to the public internet. A Gradio `share=True` tunnel is also a public endpoint, not a private local interface.

## Troubleshooting principles

### Out of memory

Reduce sequence length and batch size first. Then consider gradient accumulation, checkpointing, quantization, adapter training, or a smaller model. Record every change because it can alter quality and speed.

### Package or CUDA mismatch

Create a clean environment and verify the driver, CUDA runtime, PyTorch build, and optional attention kernels independently. Optional acceleration libraries should not be installed until the basic model can load.

### Training loss does not improve

Inspect rendered examples and labels, compare against a small overfitting test, verify the learning rate, and confirm that adapter parameters receive gradients. More steps cannot repair malformed data.

### Good benchmark, poor real use

Check leakage, prompt-template differences, retrieval quality, and whether the test set represents the actual deployment population. Add examples from observed failures only to a new development set, not retroactively to the held-out test set.

## Reproducibility checklist

- [ ] Model license and revision recorded
- [ ] Dataset provenance and split strategy documented
- [ ] Dependencies locked
- [ ] Hardware and runtime recorded
- [ ] Prompts and generation settings versioned
- [ ] Baseline evaluated before adaptation
- [ ] Held-out test set preserved
- [ ] Security and privacy boundaries tested
- [ ] Limitations and failed cases reported

## Further reading

- [Ollama documentation](https://ollama.com/download)
- [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui)
- [Hugging Face model hub](https://huggingface.co/Qwen/Qwen2-7B)

*Workflow reviewed: July 2026. Re-check upstream documentation before reproducing the commands.*
