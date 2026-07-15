---
slug: pytorch-ml-dl-tutorial
title: PyTorch for Research — A Version-Aware Learning Roadmap
description: A concise, reproducibility-focused route from tensors and training loops to transfer learning, evaluation, safe checkpoints, and deployment boundaries.
authors: [liangchao]
category: AI & machine learning
article_type: Technical guide
tags: [machine-learning, artificial-intelligence, python, computer-vision]
image: /img/blog-default.jpg
---

## Project overview

This article is a learning roadmap, not a copy-and-run production framework. It keeps one small executable example, then explains the decisions that make a research model auditable: device handling, seeds, data splits, metrics, checkpoints, version records, and validation.

- **Audience:** Python users starting reproducible machine-learning experiments
- **API scope:** recent PyTorch 2.x and torchvision releases; always check the installed-version documentation
- **Verification boundary:** examples are intentionally small and syntax-checkable; no benchmark, cloud price, or hardware-performance claim is implied

<!-- truncate -->

## 1. Install from the official selector

PyTorch packages depend on the operating system and accelerator runtime. Use the current choices at [PyTorch — Start Locally](https://pytorch.org/get-started/locally/) rather than copying a CUDA URL from an old tutorial.

A CPU-only pip installation is useful for learning and continuous integration:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

On Windows, activate the environment with the command appropriate to PowerShell or Command Prompt. For CUDA, ROCm, XPU, or another backend, use the official selector and record the exact command.

Capture the environment:

```bash
python --version
python -m pip freeze > requirements-lock.txt
```

For a maintained project, prefer a dependency file and a deliberate lock/update process over an unreviewed snapshot.

## 2. Inspect the runtime and select a device

Do not scatter unconditional `.cuda()` calls through a project. Select a device once and move both model and tensors to it.

```python
import torch

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = select_device()

print("PyTorch:", torch.__version__)
print("Device:", device)
print("CUDA runtime:", torch.version.cuda)
```

Record the device, accelerator model, driver/runtime, package versions, and precision mode with experiment results.

## 3. Set reproducibility controls before model creation

Seeds must be set before initializing a model or shuffling data.

```python
import os
import random

import numpy as np
import torch

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.use_deterministic_algorithms(True, warn_only=True)
```

A seed improves repeatability but does not guarantee identical results across PyTorch versions, devices, kernels, or distributed configurations. Report tolerances and rerun important experiments.

## 4. Train one minimal model

The XOR example demonstrates tensors, a module, logits, a loss, an optimizer, and inference without a large dataset.

```python
import torch
from torch import nn

torch.manual_seed(42)

X = torch.tensor(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
)
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for _ in range(500):
    logits = model(X)
    loss = criterion(logits, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

model.eval()
with torch.inference_mode():
    probabilities = torch.sigmoid(model(X))
    predictions = (probabilities >= 0.5).to(torch.int32)

print(probabilities.squeeze())
print(predictions.squeeze())
```

Use `BCEWithLogitsLoss` with raw logits rather than adding a sigmoid layer before `BCELoss`. Do not publish exact expected probabilities: initialization, package versions, and numerical kernels can change them.

## 5. Structure a research dataset

Keep dataset splitting independent of preprocessing fitted on the data.

1. define the observational unit;
2. split by a unit that prevents leakage, such as plant, plot, field, date, or subject;
3. fit normalization and feature transforms on the training split only;
4. freeze the validation split for model selection;
5. touch the test split only for final evaluation.

For image phenotyping, random image-level splitting can leak near-duplicate views of the same plant into training and validation. Group-aware splitting is usually more defensible.

A custom dataset should return a sample and target without silently changing global state:

```python
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

class ImageTableDataset(Dataset):
    def __init__(self, rows, transform=None):
        self.rows = list(rows)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(Path(row["path"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])
```

Validate paths and labels when constructing `rows`, and document how missing or corrupt samples are handled.

## 6. Use a transparent training loop

One epoch should have a clear contract: consume a loader, update the model, and return a sample-weighted metric.

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = 0.0
    sample_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.shape[0]
        loss_sum += loss.detach().item() * batch_size
        sample_count += batch_size

    return loss_sum / sample_count
```

For validation:

- call `model.eval()`;
- wrap inference in `torch.inference_mode()`;
- never update model parameters;
- aggregate metrics over all samples;
- store predictions and identifiers when error analysis is needed.

Avoid selecting a model from test-set performance.

## 7. Add mixed precision only when supported

Automatic mixed precision can improve CUDA throughput, but it is an optimization, not a correctness requirement. Establish a full-precision baseline first.

```python
from contextlib import nullcontext

import torch

use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

for inputs, targets in train_loader:
    inputs = inputs.to(device)
    targets = targets.to(device)
    optimizer.zero_grad(set_to_none=True)

    precision_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp
        else nullcontext()
    )

    with precision_context:
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

The old `torch.cuda.amp.autocast` and `torch.cuda.amp.GradScaler` namespaces are deprecated in current documentation. Monitor loss, gradients, and metrics for NaN or overflow when changing precision.

## 8. Start computer vision with a maintained baseline

Torchvision weight enums couple pretrained parameters to documented preprocessing:

```python
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()

model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 4)
```

Record the weights enum, input resolution, transform, class mapping, and fine-tuning policy. Avoid calling a single output head a complete YOLO detector: object detection also requires target encoding, a loss, decoding, non-maximum suppression, evaluation, and task-specific training.

For plant images, compare a learned model against simple baselines and evaluate across the domains that matter—cultivar, growth stage, sensor, field, date, and lighting.

## 9. Save state safely

Save state dictionaries and metadata rather than serializing an arbitrary live Python object:

```python
from pathlib import Path

import torch

checkpoint_path = Path("checkpoints/model.pt")
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "class_names": class_names,
    },
    checkpoint_path,
)
```

Load only trusted files and map them to a known device:

```python
checkpoint = torch.load(
    "checkpoints/model.pt",
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(checkpoint["model_state_dict"])
```

`torch.load` uses restricted loading with `weights_only=True`, but a project should still treat external model artifacts as untrusted input and verify their provenance and integrity.

## 10. Evaluate more than one number

Choose metrics before looking at final results.

- **Classification:** confusion matrix, per-class precision/recall, macro F1, calibration, and uncertainty intervals
- **Regression:** MAE, RMSE, bias, residual plots, and errors by biological or acquisition subgroup
- **Segmentation:** IoU/Dice per class, boundary error, object-level errors, and failure cases
- **Detection:** the metric definition, IoU range, object-size strata, and precision–recall curves

Report dataset composition and uncertainty. A high aggregate score can hide failure on a cultivar, field, camera, or rare class.

## 11. Treat deployment as a separate engineering phase

A notebook inference call is not a production service. Before deployment, define:

- model and preprocessing version;
- input schema, size, and content limits;
- authentication and authorization;
- request rate and resource limits;
- timeout, batching, and concurrency behavior;
- privacy and data-retention policy;
- observability, rollback, and drift monitoring;
- tests using the exported artifact, not only the training model.

ONNX, `torch.export`, TorchScript, accelerator compilers, and serving frameworks have version-specific constraints. Select one only after measuring it on the target hardware. A minimal Flask endpoint without these controls should be described as a local demonstration, not production deployment.

## Troubleshooting checklist

| Symptom          | First checks                                                                                  |
| ---------------- | --------------------------------------------------------------------------------------------- |
| Out of memory    | Input size, batch size, retained computation graphs, precision, and unused tensors            |
| Device mismatch  | Model, inputs, targets, and newly created tensors use the same device                         |
| DataLoader hangs | Start with `num_workers=0`, then increase while testing the platform                          |
| Loss is NaN      | Input ranges, labels, learning rate, loss assumptions, precision, and gradients               |
| Results change   | Seeds, data ordering, augmentation, package versions, kernels, and split leakage              |
| Checkpoint fails | Architecture and class mapping match; load a trusted state dictionary with an explicit device |

## Suggested learning sequence

1. tensors, shapes, dtypes, and autograd;
2. `Dataset`, `DataLoader`, and leakage-safe splitting;
3. one transparent training and validation loop;
4. a simple baseline and documented metrics;
5. transfer learning with versioned preprocessing;
6. experiment tracking and ablation studies;
7. export and deployment validation.

## Official resources

- [PyTorch documentation](https://pytorch.org/docs/)
- [PyTorch tutorials](https://pytorch.org/tutorials/)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [PyTorch GitHub issues](https://github.com/pytorch/pytorch/issues)
