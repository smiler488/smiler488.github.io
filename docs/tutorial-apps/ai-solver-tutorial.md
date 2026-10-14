---
title: "Multimodal AI Solver"
description: "Answer lab safety exam questions and other problems from a screen capture, camera photo or text, with structured answers, a second-answer check and a self-test of your model."
sidebar_label: "AI Solver"
sidebar_position: 12
hide_title: true
keywords:
  - "ai"
  - "solver"
  - "lab safety"
  - "exam"
  - "camera"
  - "screen"
app_route: "/app/solver"
app_icon: "Σ"
app_category: "AI & research"
app_runtime: "Uses your selected AI provider"
app_tone: "blue"
app_badges:
  - "Lab safety exam"
  - "Answer check"
  - "Camera / screen"
---

## What it does

Multimodal AI Solver sends a question to the AI model you choose and shows its answer. The question can come from:
- a selected area of your screen;
- a camera photo;
- typed text.

Its main use is **lab safety exams**. The **Lab Safety Exam** preset, selected by default:
1. returns the answer as a large letter or True/False;
2. shows the question type, the model's confidence, the question as the model read it, the reason and the rule behind it;
3. checks every answer with a second, independent answer.

Eighteen further presets cover general analysis, mathematics, lab hazard inspection, plant identification, code, translation and other subjects, and **Custom Mode** takes your own instructions.

:::info What is checked, and what is not
The site checks everything around the answer:
- screen and camera capture;
- the request format for every provider;
- reading the answer out of the model's reply;
- comparing the two answers.

These are covered by automated tests and were run end to end against a reference model. **The answer itself comes from the model you choose.** Before relying on a model, run the **Self-test** (32 lab safety questions with settled answers) to see its accuracy, and treat any question where the two answers differ as one to decide yourself.
:::

## Before you start

- Choose a provider and model under **Solver model** and enter your own API key. Screen and camera questions need a model marked "image + text".
- The provider list covers:
  - OpenAI, Anthropic, Google Gemini, xAI, Mistral and Meta;
  - DeepSeek, Qwen, Kimi, GLM, MiniMax, Doubao, Hunyuan, ERNIE, StepFun and MiMo;
  - aggregators such as OpenRouter, SiliconFlow, Groq, Together AI and Hugging Face;
  - Ollama or LM Studio on your computer;
  - any OpenAI- or Anthropic-compatible endpoint.
- Screen capture needs a desktop browser (Chrome, Edge, Firefox or Safari) and your permission to share a window. It is usually unavailable on phones; use the camera there.
- **Local demo** shows a labelled sample answer without sending anything; it does not read questions.

## Lab safety exams: quick workflow

1. [Open Multimodal AI Solver](/app/solver). **Prompt preset** shows **Lab Safety Exam**; the page remembers your choice.
2. Under **Solver model**, choose a provider, an image + text model and your API key.
3. Optional, recommended once per model: select **Run self-test** and read the score and the questions it got wrong.
4. Open the exam in another window. In the solver, choose **Screen** and select **Capture screen**. Share the exam window, drag the box over the question and its options, and select **Analyze selection**.
5. Read the answer. It appears as a short line under the buttons (for example "Answer: B · both answers agree") and as a full card under **Response**.
6. For the next question, show it in the exam window and select **Capture & ask**. Sharing stays on and the same area is captured, so each question is one click. Use **Adjust area** if the question moves, and **Stop sharing** when you finish.

Typed questions work the same way under **Text**: paste the question with its options and select **Ask**.

## Reading the answer

| Item | Meaning |
| --- | --- |
| Large answer | Option letters (one for single choice, several for multiple choice), True/False, or text for fill-in and short answers. |
| Type · confidence | Question type and the model's own confidence (high, medium, low). |
| **A second, independent answer agrees** | A second request, asked to judge each option separately before answering, gave the same answer. |
| **The second answer differs (…)** | The two requests disagree: the question is ambiguous or the model is unsure. Read it again and decide yourself. |
| **Question as read** | What the model read from the capture. If it is not the question on screen, adjust the area. |
| Reason, **Basis** | The model's explanation and the safety rule it applies. |
| **Full model reply** | The raw reply, for checking. |

## Self-test

**Run self-test** sends 32 lab safety questions (20 single choice, 4 multiple choice, 8 true/false) with settled answers to the selected model, one at a time, without the second-answer check. It reports how many it answered correctly and lists each miss with the key and the model's answer. Each question is one request to your provider, so a run costs 32 requests.

The questions cover dilution of acids, metal fires, electrical fires, chemical burns and eye exposure, fume hoods, flammable storage, gas cylinders, mercury spills, waste, centrifuges, electric shock, biosafety cabinets, infectious waste, highly toxic chemicals, extinguishers, evacuation, personal protection and autoclaves. A model that scores below about 90% here should not be trusted on its own in an exam.

## Controls & outputs

| Control | Purpose |
| --- | --- |
| **Camera / Screen / Text** | Chooses the input. |
| **Enable camera**, **Capture and solve** | Starts the camera and sends one photo, at most 1,280 px on its longest side. |
| **Capture screen** | Asks to share a screen or window, captures it and lets you place the selection box. |
| **Analyze selection** | Sends the selected area (PNG for normal question sizes, so small text stays sharp). |
| **Capture & ask**, **Adjust area**, **Stop sharing** | One-click capture of the same area while sharing stays on; move the area; end sharing. |
| **Ask** | Sends the typed text. |
| **Prompt preset** | Lab Safety Exam, 18 other presets or Custom Mode. Presets now apply in text mode too. |
| Second-answer check | On by default for Lab Safety Exam; turning it off halves the requests. |
| **Run self-test** | Measures the selected model on the question bank. |

## Data, privacy & external services

- The camera picture and the screen stay on your device until you send a question; only the selected area or photo and the prompt go to the endpoint shown under **Solver model**.
- The API key stays in the tab's memory and is cleared on refresh, provider change or leaving the page.
- Do not capture passwords, private messages or other content you are not allowed to send to the provider.

:::caution Your responsibility
Follow the rules of the exam you take. Safety rules differ between institutions; where the model's answer conflicts with your institution's training material, the material is right.
:::

## Limitations

- Answers depend on the model; the self-test measures one model on one set of questions, not every possible question.
- A question split across two screens, or an image with several questions, is answered for the first complete question only.
- There is no response history or export.

## Troubleshooting

- **"Question as read" is wrong or incomplete:** select **Adjust area** and include the whole question and all options.
- **The two answers often differ:** use a stronger model, or check the self-test score.
- **The model rejects images:** choose a model marked "image + text", or type the question.
- **Screen capture is unavailable:** use a desktop browser and allow screen sharing; on phones use the camera.
- **401, 403, 404 or 429:** check the provider, key, model ID, permissions and quota.
- **Could not reach the provider:** the provider blocks direct browser access; choose another provider or use your own proxy.

[Open Multimodal AI Solver →](/app/solver)

[← Back to App Lab](/app)
