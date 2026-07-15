---
title: "Multimodal AI Solver"
description: "Ask with text, camera or screen capture and send the prompt to the visual AI model and API provider you configure."
sidebar_label: "AI Solver"
sidebar_position: 12
hide_title: true
keywords:
  - "ai"
  - "solver"
  - "camera"
  - "screen"
  - "vision"
  - "text"
app_route: "/app/solver"
app_icon: "Σ"
app_category: "AI & research"
app_runtime: "Uses your selected AI provider"
app_tone: "blue"
app_badges:
  - "BYOK AI"
  - "Camera / screen"
  - "Preset prompts"
---

## What it does

Multimodal AI Solver sends a text question, camera frame or selected screen region to the AI provider and model you configure. It includes 18 predefined image-analysis prompt templates plus **Custom Mode**. The default **Local demo** exercises the interface without a network request.

:::caution Local demo does not inspect pixels
Local demo returns a sample response and does not identify or reason about the captured image. Use a live, vision-capable model for actual camera or screen analysis.
:::

## Before you start

- Camera and screen capture require a recent browser, a secure HTTPS context and explicit permission.
- Screen capture availability varies by browser and device and is often limited on mobile.
- Live camera or screen analysis requires a model that accepts image input.
- Live requests require your own provider key and network access.
- Never capture passwords, private messages, unpublished data or regulated records unless sending them to the selected provider is authorized.

## Quick workflow

1. [Open Multimodal AI Solver](/app/solver).
2. Under **Input Mode**, select one workflow:
   - **Camera Capture:** select **Enable camera**, grant permission, then select **Capture and solve**.
   - **Screen Capture:** select **Capture Screen**, choose a screen, window or tab, adjust the selection box, then select **Analyze Selected Area**.
   - **Text Question:** enter text and select **Ask and Get Answer**.
3. Under **Solver model**, keep Local demo or choose a provider, model and API key. Camera and screen modes display a warning when the selected preset is not marked as vision-capable.
4. Under **Prompt Settings**, choose a **PromptPreset**. Select **Custom Mode** to edit **Question (for image mode)**.
5. Read the result under **Response**. If the provider supplies usage metadata, token counts appear at the end.

## Controls & outputs

| Control or output                   | Purpose                                                                             |
| ----------------------------------- | ----------------------------------------------------------------------------------- |
| **Enable camera / Turn off camera** | Starts or stops the local video stream. The page never starts it automatically.     |
| **Capture and solve**               | Captures one compressed camera frame and submits it with the current image prompt.  |
| **Capture Screen**                  | Requests browser screen-sharing permission and captures one frame.                  |
| Selection box                       | Moves or resizes the screen crop with four corner and four edge handles.            |
| **Analyze Selected Area**           | Encodes and submits only the selected screen region.                                |
| **Ask and Get Answer**              | Sends the text field exactly as entered.                                            |
| **Solver model**                    | Selects Local demo or a configured external provider, endpoint, model and key.      |
| **PromptPreset**                    | Selects one of 18 predefined prompts or Custom Mode for camera and screen requests. |
| **Last image size**                 | Reports the encoded JPEG size after an image is prepared.                           |
| **Response**                        | Displays provider text and final token usage when available.                        |

The predefined prompts cover general analysis, mathematics, laboratory safety, plant identification, code review, academic literature translation, physics, chemistry, English learning, text extraction, biology, history, medical images, engineering drawings, financial statements, art, legal documents and educational material.

### Current text-preset behavior

PromptPreset instructions are currently attached to camera and screen requests only. A normal **Text Question** is sent without the selected preset prompt.

Entering an exact command such as `/preset Math Problem Solver` changes the image-mode preset and returns a confirmation message. The command itself does not run an analysis, and the next ordinary text question is still sent without that preset prompt.

## How it works

Camera capture scales a frame so its longest side is no more than 1,280 pixels, then encodes it as JPEG at quality 0.85. The live preview remains on the device; the encoded still image is created only after **Capture and solve**.

Screen mode asks the browser for a shared display stream, captures one frame and immediately stops the stream. The frame stays local while you move or resize the selection. **Analyze Selected Area** crops the selected source pixels and encodes them as JPEG at quality 0.85.

For image modes, the app sends the JPEG as base64 together with the current image prompt. Text mode sends only the entered text. Starting a new request aborts any previous in-progress request managed by the page.

The app displays returned text as-is. Token usage is appended only when the provider response includes recognizable usage fields.

## Data, privacy & external services

Camera video and an unsubmitted screen frame remain local. Selecting **Capture and solve** or **Analyze Selected Area** sends the resulting image and prompt to the endpoint shown in **Solver model**. Selecting **Ask and Get Answer** sends the entered text.

The API key is held only in the current tab's memory, cleared on provider change, refresh or page exit, and sent to the displayed endpoint. Browser-entered credentials are not protected like server-side secrets. Use a restricted test key; use your own authenticated backend proxy for production.

Direct requests can fail because of provider CORS policy, regional restrictions, model permissions, billing or quota even when the key is correct.

:::caution Human verification is required
Presets are prompt templates, not validated expert systems. Medical, legal, financial, laboratory-safety and identification outputs may be wrong and must not replace qualified professional review.
:::

## Limitations

- There is no image-file upload, response history, copy button, download or share workflow.
- Local demo does not analyze image content.
- Text questions do not currently inherit PromptPreset instructions.
- Presets guide a model but do not add specialist databases, deterministic OCR or verification.
- Image quality and answer quality depend on the device, crop, provider and exact model.
- Screen capture may be unavailable on mobile or restricted by browser policy.
- A custom model ID may support vision even when the app cannot verify that capability.
- Token counts appear only when returned by the provider.

## Troubleshooting

- **Camera is off:** select **Enable camera** and grant permission; close other software using the camera.
- **Camera access is not supported:** use a recent browser over HTTPS or switch to Text Question.
- **Screen capture is unavailable or cancelled:** check browser/device support and grant the browser's share permission.
- **The selection cannot shrink further:** its minimum dimension is 50 source pixels, or the source frame size when smaller.
- **The model rejects an image:** choose a vision-capable model or use Text Question with a text-only model.
- **A preset command is not found:** use lowercase `/preset ` followed by the exact preset ID or full displayed name.
- **401, 403, 404 or 429:** verify provider, key, exact model ID, permissions, billing and quota.
- **CORS or Failed to fetch:** the provider blocks direct browser access; use an authenticated backend proxy.
- **The response is empty or inaccurate:** improve the crop and prompt, retry with a suitable model, and verify the result independently.

[Open Multimodal AI Solver →](/app/solver)

[← Back to App Lab](/app)
