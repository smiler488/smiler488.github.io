---
slug: botanical-extract-ai-pro
title: "Botanical Extract AI Pro: An Experimental Background-Isolation Workflow"
authors: [liangchao]
category: AI & machine learning
article_type: Workflow
tags: [artificial-intelligence, computer-vision, image-analysis, plant-phenotyping]
image: /img/botanical-extract-ai-pro.png
description: An experimental web and batch workflow for placing plants on white backgrounds with multimodal image models, including scientific and privacy limitations.
---

## Overview

![Examples of source plant images and AI-generated white-background outputs](/img/botanical-extract-ai-pro.png)

**Botanical Extract AI Pro** explores whether a multimodal image model can reduce background clutter in plant photographs without project-specific model training. It provides an interactive web workflow for individual images and a Node.js-oriented workflow for repeated processing.

This is best described as **AI-assisted background isolation**, not validated scientific segmentation. A generative model may redraw leaves, remove thin structures, change colors, or invent boundaries even when the prompt asks it not to.

<!-- truncate -->

:::warning Experimental output
“Zero-shot” means that no task-specific training was performed. It does not imply zero error, pixel-perfect preservation, or suitability for quantitative phenotyping.
:::

## Workflow

The two clients follow the same basic path:

1. Load an image and decode it to confirm the format and dimensions.
2. Select the closest aspect ratio supported by the chosen model API.
3. Send the image and a constrained background-replacement instruction to the provider.
4. Save the returned image beside the untouched original.
5. Record the provider, model, prompt version, time, and processing outcome.

The web interface supports visual comparison for a single sample. The batch workflow scans a directory, mirrors its structure in an output directory, and records successes and failures.

## What the prompt can and cannot do

A prompt can request a white background and ask the model to preserve the plant. It cannot force a generative system to retain the original pixels. Statements such as “pixel-perfect” are therefore **instructions to the model**, not guarantees about the result.

Aspect-ratio matching is similarly limited. Choosing the nearest supported ratio can reduce avoidable cropping or stretching in the API request, but it does not guarantee identical geometry or resolution in the generated image.

## Input validation

File extensions alone are not sufficient validation. A safer client should:

- Check the binary signature and successfully decode the image
- Enforce file-size and pixel-dimension limits
- Reject unsupported or malformed data
- Normalize orientation without silently discarding the original
- Keep provider payloads separate from local metadata

The current Base64-style request pattern loads the image into memory. It should not be described as streaming, and large inputs need explicit limits.

## Bounded batch processing

`Promise.allSettled(files.map(processFile))` does **not** control concurrency; it schedules every task immediately. A production batch client should place a small limit around the provider call:

```javascript
import pLimit from 'p-limit';

const limit = pLimit(3);

const results = await Promise.allSettled(
  files.map((file) => limit(() => processOneImage(file))),
);
```

The processing function should also use capped exponential backoff for retryable `429` and `5xx` responses, stop retrying permanent errors, and write a resumable manifest. Concurrency must be adjusted to the provider's documented limits and the available memory.

## Scientific quality control

Generated outputs should not be used directly for leaf area, shape, disease, growth, or time-series measurements without validation. A practical review should include:

1. Overlay the source and output at the same scale.
2. Inspect thin stems, leaf tips, holes, flowers, labels, and pot boundaries.
3. Check whether colors, shadows, or plant geometry changed.
4. Compare a manually annotated validation subset with appropriate mask and boundary metrics.
5. Reject or manually correct failed samples before quantitative analysis.

For measurements that require a reproducible binary mask, a conventional segmentation model or a validated interactive segmentation tool is usually more appropriate than image generation.

## Privacy, cost, and reproducibility

Unless a local model is used, uploaded images leave the device and are processed by an external provider. Before processing research data:

- Review the provider's retention and training policies.
- Remove sensitive labels, locations, and personal information.
- Store API keys in server-side or environment configuration, never in public client code.
- Estimate per-image cost and rate limits before starting a batch.
- Preserve the original files and a machine-readable processing manifest.

## Current limitations

- No public benchmark currently establishes accuracy across species, organs, backgrounds, or imaging conditions.
- Results may vary across model versions and repeated requests.
- The workflow produces an edited image, not necessarily an alpha mask or class-labeled segmentation.
- Large-scale processing requires bounded concurrency, recovery logs, and manual quality assurance.
- The project should not be described as open source unless a public repository and license are provided.

The project remains useful as a prototype for rapid visual cleanup and for studying how multimodal models behave on botanical imagery. Its outputs should be treated as generated derivatives, not as ground truth.
