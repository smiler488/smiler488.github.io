import { useCallback, useEffect, useRef, useState } from "react";
import AppScaffold from "../../../components/AppScaffold";
import Heading from "@theme/Heading";
import CitationNotice from "../../../components/CitationNotice";
import AIProviderSettings from "../../../components/AIProviderSettings";
import {
  createDefaultAIConfig,
  getAIProvider,
  requestAI,
} from "../../../lib/api";
import {
  answerLabel,
  examPrompt,
  examSystemPrompt,
  matchesKey,
  parseExamAnswer,
  sameAnswer,
} from "../../../lib/solver/exam";
import styles from "./styles.module.css";
import { IS_ZH, makeToolText } from "@site/src/lib/i18n/toolText";
import ZH from "./_zh";

const tx = makeToolText(ZH);

const EXAM_PRESET = "lab_exam";
const PRESET_STORAGE_KEY = "solver:preset";
const SELFTEST_PATH = "/app/solver/lab-safety-selftest.json";

const TYPE_LABELS = {
  single: tx("Single choice"),
  multiple: tx("Multiple choice"),
  true_false: tx("True / false"),
  fill: tx("Fill in the blank"),
  short: tx("Short answer"),
};

const CONFIDENCE_LABELS = {
  high: tx("High confidence"),
  medium: tx("Medium confidence"),
  low: tx("Low confidence"),
};

function readStoredPreset() {
  try {
    return window.localStorage.getItem(PRESET_STORAGE_KEY);
  } catch {
    return null;
  }
}

function storePreset(id) {
  try {
    window.localStorage.setItem(PRESET_STORAGE_KEY, id);
  } catch {
    // Storage blocked: the choice is simply not remembered.
  }
}

// Self-test question as text, the way it appears in an exam.
function selfTestText(q) {
  if (q.type === "true_false") {
    return `判断题：${q.question}（正确还是错误？）`;
  }
  const kind = q.type === "multiple" ? "多选题" : "单选题";
  const options = Object.entries(q.options)
    .map(([k, v]) => `${k}. ${v}`)
    .join("\n");
  return `${kind}：${q.question}\n${options}`;
}

const INPUT_MODES = [
  { id: "camera", label: tx("Camera") },
  { id: "screenshot", label: tx("Screen") },
  { id: "text", label: tx("Text") },
];

function useCamera() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const requestIdRef = useRef(0);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState(null);
  const [starting, setStarting] = useState(false);

  const stopCamera = useCallback(() => {
    requestIdRef.current += 1;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setStarting(false);
    setReady(false);
  }, []);

  const startCamera = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError(tx("Camera access is not supported by this browser."));
      return;
    }
    stopCamera();
    const requestId = requestIdRef.current;
    setError(null);
    setStarting(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });
      if (requestId !== requestIdRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setReady(true);
      }
    } catch (cameraError) {
      setError(cameraError?.message || String(cameraError));
    } finally {
      if (requestId === requestIdRef.current) setStarting(false);
    }
  }, [stopCamera]);

  useEffect(() => stopCamera, [stopCamera]);

  return { videoRef, ready, error, starting, startCamera, stopCamera };
}

async function captureCompressedJpeg(video, maxSide = 1280, quality = 0.85) {
  const w = video.videoWidth || 1080;
  const h = video.videoHeight || 1440;
  const scale = Math.min(1, maxSide / Math.max(w, h));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(w * scale);
  canvas.height = Math.round(h * scale);
  const ctx = canvas.getContext("2d");
  if (!ctx)
    throw new Error(tx("Canvas rendering is not available in this browser."));
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise((resolve) =>
    canvas.toBlob(resolve, "image/jpeg", quality)
  );
  if (!blob)
    throw new Error(tx("The browser could not encode the captured image."));
  return blob;
}

function formatAiResponse(result) {
  const text = result?.text || "The provider returned an empty response.";
  const usage = result?.usage || {};
  const promptTokens =
    usage.prompt_tokens ??
    usage.input_tokens ??
    usage.promptTokenCount ??
    usage.PromptTokens;
  const completionTokens =
    usage.completion_tokens ??
    usage.output_tokens ??
    usage.candidatesTokenCount ??
    usage.CompletionTokens;
  const totalTokens =
    usage.total_tokens ??
    usage.totalTokenCount ??
    usage.TotalTokens ??
    (typeof promptTokens === "number" && typeof completionTokens === "number"
      ? promptTokens + completionTokens
      : null);

  if (promptTokens == null && completionTokens == null && totalTokens == null) {
    return text;
  }

  return `${text}\n\n---\nTokens: ${totalTokens ?? "N/A"} (Prompt: ${
    promptTokens ?? "N/A"
  }, Completion: ${completionTokens ?? "N/A"})`;
}

function getSafeAiErrorMessage(error) {
  const message = error instanceof Error ? error.message : String(error || "");
  const normalized = message.toLowerCase();

  if (
    normalized.includes("enter an api endpoint") ||
    normalized.includes("choose or enter a model") ||
    normalized.includes("enter the api key")
  ) {
    return message;
  }
  if (
    /\b401\b|unauthori[sz]ed|authentication|invalid api key/.test(normalized)
  ) {
    return "Authentication failed. Check the API key for the selected provider.";
  }
  if (/\b403\b|permission|forbidden/.test(normalized)) {
    return "The provider denied this request. Check key permissions, billing, and model access.";
  }
  if (/\b404\b|not found/.test(normalized)) {
    return "The API endpoint or model was not found. Check the provider settings.";
  }
  if (/\b429\b|rate limit|too many requests/.test(normalized)) {
    return "The provider rate limit was reached. Wait briefly or check the account quota.";
  }
  if (/cors|failed to fetch|network|load failed|connection/.test(normalized)) {
    return "The browser could not reach the provider. Check the endpoint, connection, and CORS support.";
  }
  if (/\b5\d\d\b|service unavailable|overloaded/.test(normalized)) {
    return "The provider is temporarily unavailable. Please try again later.";
  }
  if (/\b400\b|bad request|invalid request/.test(normalized)) {
    return "The provider rejected the request. Check that the selected model supports this input type.";
  }

  return "The AI request failed. Check the provider, model, endpoint, and account status.";
}

export default function SolverAppPage() {
  const { videoRef, ready, error, starting, startCamera, stopCamera } =
    useCamera();

  // State management
  const [aiConfig, setAiConfig] = useState(createDefaultAIConfig);
  // Preset prompt list
  // Preset prompt list — full English version
  const promptPresets = [
    {
      id: EXAM_PRESET,
      name: tx("Lab Safety Exam"),
      prompt: "",
      description: tx(
        "Answers lab safety exam questions (single choice, multiple choice, true/false) with the reason and the rule behind it, and checks each answer with a second independent answer."
      ),
    },
    {
      id: "default",
      name: tx("Intelligent Analysis Assistant"),
      prompt: `ROLE: Professional multimodal analyst
TASK: Analyze the problem shown in the image, photo, or text, and provide a practical, step-by-step solution.
METHOD:
- If an image is provided: briefly describe the visible content and extract key information (titles, labels, warnings, numbers).
- Identify the core problem and its constraints, then decompose it into sub-questions.
- Provide the optimal solution path; if multiple approaches exist, show the top 2 with pros and cons.
OUTPUT:
- Understanding (key issue and input summary)
- Steps/Analysis (reasoning and procedure)
- Answer (final actionable solution and parameters)
- Checks/Uncertainties (assumptions and unclear areas)
- Next actions (1–3 follow-up suggestions)`,
      description: tx("General intelligent analysis and solution generation"),
    },

    {
      id: "math",
      name: tx("Math Problem Solver"),
      prompt: `ROLE: Mathematician and educator
TASK: Solve the given math problem with rigorous reasoning and precise results.
METHOD:
- List knowns, unknowns, conditions, and goals; unify notations and units.
- Choose the most elegant, rigorous method; optionally include one alternative.
- Keep derivations auditable; show analytical expressions before numeric results.
OUTPUT:
- Understanding (knowns, goal, notation)
- Steps/Analysis (key formulas and derivation)
- Answer (final result: exact form → numeric)
- Checks/Uncertainties (domain restrictions, special cases)
- Next actions (numerical verification or plotting tip)`,
      description: tx(
        "Equation solving, geometry, algebra, and other math problems"
      ),
    },

    {
      id: "lab_safety",
      name: tx("Laboratory Safety Assessment"),
      prompt: `ROLE: Laboratory safety inspector
TASK: Identify potential hazards and compliance issues in the lab image/text and recommend corrections.
METHOD:
- Classify risks: chemical, biological, physical, electrical, ergonomic, fire, etc.
- Identify issues based on general safety standards (PPE, labeling, segregation, ventilation, waste disposal).
- Provide corrective measures with priorities (high/medium/low).
OUTPUT:
- Understanding (scene and visible risks)
- Findings by category (issues + priority)
- Recommendations (specific corrective steps and PPE list)
- Checks/Uncertainties (unverifiable elements)
- Next actions (training, documentation, inspection cycle)`,
      description: tx(
        "Laboratory safety risk assessment and improvement suggestions"
      ),
    },

    {
      id: "plant",
      name: tx("Plant Identification Analysis"),
      prompt: `ROLE: Botanist
TASK: Identify the plant in the image and provide care and usage information.
METHOD:
- Describe diagnostic traits (leaf, flower, fruit, bark, growth habit, habitat).
- Provide scientific and common names, plus 2–3 lookalikes with distinguishing features.
- Include native range, growth conditions, and care tips; note toxicity if relevant.
- If uncertain, list top-3 candidates with confidence percentages.
OUTPUT:
- Understanding (visible traits)
- Candidate identification(s) with confidence
- Care and uses (cultivation and application)
- Checks/Uncertainties (missing angles/phenological stage)
- Next actions (additional photos or observation tips)`,
      description: tx("Plant species identification and growth information"),
    },

    {
      id: "code",
      name: tx("Code Review and Optimization"),
      prompt: `ROLE: Senior software engineer
TASK: Explain the purpose of the code, find potential bugs or inefficiencies, and provide improved code.
METHOD:
- Detect language/environment, summarize purpose and data flow.
- Identify logic/syntax errors, boundary cases, and performance bottlenecks.
- Provide corrected, runnable code and a minimal reproducible example with basic test inputs.
OUTPUT:
- Understanding (purpose, I/O, environment)
- Issues (bugs, inefficiencies, and reasoning)
- Improved code (ready-to-run version)
- Complexity (Big-O time and space)
- Checks/Uncertainties (dependencies, assumptions)
- Next actions (testing, linting, profiling)`,
      description: tx(
        "Code explanation, debugging, and performance optimization"
      ),
    },

    {
      id: "translation",
      name: tx("Academic Literature Translation"),
      prompt: `ROLE: Academic translator/editor
TASK: Translate the academic text into the target language with accurate terminology and fluent scholarly style.
METHOD:
- Preserve structure (headings, lists, tables, citations).
- Create a short glossary for key technical terms; add minimal notes only if necessary.
OUTPUT:
- Understanding (domain and tone)
- Translation (publication-quality output)
- Glossary (5–15 essential terms)
- Notes (only if needed)`,
      description: tx(
        "Professional academic translation and terminology explanation"
      ),
    },

    {
      id: "physics",
      name: tx("Physics Problem Solver"),
      prompt: `ROLE: Physicist
TASK: Solve the physics problem step-by-step using fundamental laws and correct units.
METHOD:
- List knowns/unknowns, define the system boundaries, and state the governing principles (Newton, conservation, Maxwell, thermodynamics, etc.).
- Show clear derivations, ensure dimensional consistency, and estimate experimental uncertainties.
OUTPUT:
- Understanding (variables and setup)
- Steps/Analysis (laws and reasoning)
- Answer (result with units and significant figures)
- Checks/Uncertainties (error sources, limiting cases)
- Next actions (experimental verification idea)`,
      description: tx(
        "Mechanics, electromagnetism, thermodynamics, and other physics problems"
      ),
    },

    {
      id: "chemistry",
      name: tx("Chemistry Problem Solver"),
      prompt: `ROLE: Chemist
TASK: Analyze chemical reactions, stoichiometry, structures, or lab procedures.
METHOD:
- Identify species and conditions; write and balance reactions; outline mechanisms when relevant.
- For lab operations: discuss safety, feasibility, yields, and waste disposal (general guidance).
OUTPUT:
- Understanding (species, conditions, target)
- Steps/Analysis (equations, mechanisms, calculations)
- Answer (balanced results, quantities, or expected phenomena)
- Checks/Uncertainties (side reactions, sensitivity)
- Next actions (controls, optimization suggestions)`,
      description: tx(
        "Chemical equations, structures, mechanisms, and experiment analysis"
      ),
    },

    {
      id: "english",
      name: tx("English Learning Assistant"),
      prompt: `ROLE: English tutor
TASK: Analyze English text, explain grammar and vocabulary, and provide detailed exercise solutions.
METHOD:
- Break down syntax (clauses, parts of speech, collocations).
- Give clear explanations and 1–2 natural example sentences for difficult expressions.
OUTPUT:
- Understanding (topic and level)
- Analysis (grammar and vocabulary)
- Answers with explanations (if exercises)
- Next actions (3–5 mini practice items)`,
      description: tx(
        "Grammar explanation, vocabulary learning, and exercise guidance"
      ),
    },

    {
      id: "ocr",
      name: tx("Text Extraction"),
      prompt: `ROLE: OCR transcriber and formatter
TASK: Extract all text from the image accurately while preserving layout and structure.
METHOD:
- Maintain original paragraphing and headings; reconstruct tables using Markdown tables; preserve math/code blocks.
- Use [?] markers for uncertain words and list them separately.
OUTPUT:
- Understanding (source type and readability)
- Extracted text (structured)
- Uncertain fragments (list with notes)
- Next actions (reshoot tips: angle, resolution, key region)`,
      description: tx(
        "Accurate image text recognition and format preservation"
      ),
    },

    {
      id: "biology",
      name: tx("Biology Analysis"),
      prompt: `ROLE: Biology instructor
TASK: Identify and explain biological structures or processes in the image.
METHOD:
- For anatomy: label parts and functions; for cellular/molecular: describe pathways; for ecology: outline species relationships.
- Present explanations precisely in descriptive text form.
OUTPUT:
- Understanding (context and scale)
- Analysis (structures/processes and relationships)
- Findings (main biological insights)
- Checks/Uncertainties
- Next actions (observation or experiment suggestion)`,
      description: tx(
        "Biological structures, cell biology, and ecology analysis"
      ),
    },

    {
      id: "history",
      name: tx("Historical Document Interpretation"),
      prompt: `ROLE: Historian
TASK: Interpret the document or artifact’s background, era, and cultural significance.
METHOD:
- Describe visible features; infer materials and craftsmanship; relate to known historical periods (no fabricated citations).
OUTPUT:
- Understanding (object or text summary)
- Analysis (era, craftsmanship, context)
- Findings (historical significance)
- Checks/Uncertainties
- Next actions (evidence needed for dating/provenance)`,
      description: tx("Historical document or artifact interpretation"),
    },

    {
      id: "medical",
      name: tx("Medical Image Analysis"),
      prompt: `ROLE: Medical information assistant (not a clinician)
TASK: Identify anatomical structures and observable findings in medical images and explain their general significance.
DISCLAIMER: This analysis is for informational purposes only and does NOT replace professional medical advice.
METHOD:
- Describe the modality and view; list findings in structured form; provide possible interpretations without diagnosis.
OUTPUT:
- Understanding (modality, region)
- Findings (structured observations)
- Significance (general clinical meaning)
- Checks/Uncertainties
- Next actions (what to discuss with a doctor)`,
      description: tx(
        "Medical image structure and general interpretation (non-diagnostic)"
      ),
    },

    {
      id: "engineering",
      name: tx("Engineering Drawing Interpretation"),
      prompt: `ROLE: Engineering analyst
TASK: Interpret the engineering drawing or chart and explain its design intent and parameters.
METHOD:
- Identify drawing type (mechanical, civil, circuit, P&ID, etc.); decode symbols and summarize key dimensions or tolerances.
- For circuits: signal flow and core components; for mechanical: parts, fits, materials, and notes.
OUTPUT:
- Understanding (type and scope)
- Analysis (components, parameters, relationships)
- Findings (key tolerances, critical paths)
- Checks/Uncertainties
- Next actions (manufacture, assembly, validation)`,
      description: tx(
        "Engineering drawings, technical charts, and design interpretation"
      ),
    },

    {
      id: "finance",
      name: tx("Financial Statement Analysis"),
      prompt: `ROLE: Financial analyst (informational only)
TASK: Analyze financial statements or data for performance, trends, and risks.
METHOD:
- Compute key metrics (growth, margins, ROE/ROA, leverage, liquidity) based on provided data only.
- Interpret results concisely and note 2–3 main risks.
OUTPUT:
- Understanding (scope and period)
- Metrics (summary table or bullets)
- Analysis (drivers of performance)
- Findings (strengths and risks)
- Checks/Uncertainties
- Next actions (questions for management, monitoring points)`,
      description: tx(
        "Financial statement interpretation and business performance analysis"
      ),
    },

    {
      id: "art",
      name: tx("Artwork Appreciation"),
      prompt: `ROLE: Art critic
TASK: Analyze the artwork’s style, technique, composition, and aesthetic value.
METHOD:
- Discuss composition, perspective, color, texture, medium, and artistic influences; do not invent provenance.
OUTPUT:
- Understanding (subject and medium)
- Analysis (style, technique, composition)
- Findings (aesthetic and contextual meaning)
- Checks/Uncertainties
- Next actions (comparative references or similar works)`,
      description: tx("Art style, technique, and aesthetic value analysis"),
    },

    {
      id: "legal",
      name: tx("Legal Document Interpretation"),
      prompt: `ROLE: Legal information assistant (not a lawyer)
TASK: Explain rights, obligations, and risk points in the legal document.
DISCLAIMER: This content is for general informational purposes and not legal advice.
METHOD:
- Summarize parties, term, key clauses (payment, IP, termination, liability, confidentiality, dispute resolution), and red flags.
OUTPUT:
- Understanding (document scope)
- Clause highlights
- Risks and ambiguities
- Checks/Uncertainties
- Next actions (questions or negotiable clauses)`,
      description: tx(
        "Legal document and contract clause interpretation (non-advisory)"
      ),
    },

    {
      id: "education",
      name: tx("Educational Material Analysis"),
      prompt: `ROLE: Instructional designer
TASK: Analyze teaching materials for educational value, target audience, and pedagogy.
METHOD:
- Identify learning objectives, prerequisites, key concepts, and assessment focus.
- Suggest teaching activities, sequencing, and improvement points.
OUTPUT:
- Understanding (topic and learners)
- Analysis (objectives and knowledge map)
- Findings (strengths and gaps)
- Next actions (activities, assessments, improvements)`,
      description: tx(
        "Educational material evaluation and teaching method design"
      ),
    },

    {
      id: "custom",
      name: tx("Custom Mode"),
      prompt: `ROLE: Flexible expert
TASK: Handle special or mixed requests not covered by other presets, following general reasoning rules.
METHOD:
- Clarify objective and assumptions; provide an immediately usable solution with alternative paths if needed.
OUTPUT:
- Understanding
- Steps/Analysis
- Answer / Deliverable
- Checks/Uncertainties
- Next actions`,
      description: tx("Flexible mode for custom or mixed scenarios"),
    },
  ];

  const [selectedPreset, setSelectedPreset] = useState(EXAM_PRESET);
  const [question, setQuestion] = useState("");
  const [examResult, setExamResult] = useState(null);
  const [doubleCheck, setDoubleCheck] = useState(true);
  const [sharing, setSharing] = useState(false);
  const [hasArea, setHasArea] = useState(false);
  const shareRef = useRef(null);
  const lastBoxRef = useRef(null);
  const [selfTest, setSelfTest] = useState(null);
  const selfTestAbortRef = useRef(null);
  const isExam = selectedPreset === EXAM_PRESET;
  const usesModel = getAIProvider(aiConfig.provider).protocol !== "mock";
  const [respText, setRespText] = useState("");
  const [busy, setBusy] = useState(false);
  const [lastSizeKB, setLastSizeKB] = useState(null);
  const [captureMode, setCaptureMode] = useState("camera"); // 'camera', 'screenshot', 'text'
  const [textInput, setTextInput] = useState(""); // Text input
  const [screenshotData, setScreenshotData] = useState(null); // Screenshot data
  const [selectionBox, setSelectionBox] = useState(null); // Selection box
  const [isSelecting, setIsSelecting] = useState(false); // Whether selecting
  const screenshotViewportRef = useRef(null);
  const dragCleanupRef = useRef(null);
  const requestControllerRef = useRef(null);

  useEffect(() => {
    if (captureMode !== "camera") stopCamera();
  }, [captureMode, stopCamera]);

  useEffect(() => {
    const stored = readStoredPreset();
    if (stored && promptPresets.some((p) => p.id === stored)) {
      choosePreset(stored);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function choosePreset(id) {
    setSelectedPreset(id);
    storePreset(id);
    setExamResult(null);
    const preset = promptPresets.find((p) => p.id === id);
    if (preset) setQuestion(preset.id === "custom" ? "" : preset.prompt);
  }

  useEffect(
    () => () => {
      dragCleanupRef.current?.();
      requestControllerRef.current?.abort();
      selfTestAbortRef.current?.abort();
      shareRef.current?.stream.getTracks().forEach((track) => track.stop());
    },
    []
  );

  // Lab safety exam: a structured answer, optionally checked by a second
  // independent answer (asked to weigh each option first).
  async function askExam({ text = "", imageBase64 = null }, signal) {
    const ask = (variant) =>
      requestAI(
        aiConfig,
        { question: examPrompt({ text, variant }), imageBase64 },
        {
          mockTag: "ai-solver",
          jsonMode: true,
          maxTokens: 1200,
          systemPrompt: examSystemPrompt(IS_ZH),
          requireVision: Boolean(imageBase64),
          signal,
        }
      );
    const checking = doubleCheck && usesModel;
    const [main, check] = await Promise.allSettled([
      ask("main"),
      checking ? ask("check") : Promise.resolve(null),
    ]);
    if (main.status === "rejected") throw main.reason;
    const mainAnswer = parseExamAnswer(main.value.text);
    const checkAnswer =
      check.status === "fulfilled" && check.value
        ? parseExamAnswer(check.value.text)
        : null;
    return {
      main: mainAnswer,
      check: checkAnswer,
      checkFailed: checking && !checkAnswer,
      agree:
        mainAnswer && checkAnswer
          ? sameAnswer(mainAnswer.answer, checkAnswer.answer)
          : null,
      raw: main.value,
    };
  }

  // 通用的发送到AI的函数
  async function sendToAI(payload) {
    requestControllerRef.current?.abort();
    const controller = new AbortController();
    requestControllerRef.current = controller;
    setExamResult(null);
    try {
      if (isExam) {
        const result = await askExam(
          { text: payload.text || "", imageBase64: payload.imageBase64 },
          controller.signal
        );
        setExamResult(result);
        setRespText(formatAiResponse(result.raw));
        return;
      }
      const preset = promptPresets.find((p) => p.id === selectedPreset);
      const result = await requestAI(
        aiConfig,
        payload.imageBase64
          ? { imageBase64: payload.imageBase64, question }
          : { question: payload.text },
        {
          mockTag: "ai-solver",
          requireVision: Boolean(payload.imageBase64),
          // Text mode applies the preset as the system instructions.
          systemPrompt:
            !payload.imageBase64 && preset?.prompt ? preset.prompt : undefined,
          signal: controller.signal,
        }
      );
      setRespText(formatAiResponse(result));
    } catch (error) {
      if (error?.name === "AbortError") return;
      throw new Error(getSafeAiErrorMessage(error));
    } finally {
      if (requestControllerRef.current === controller)
        requestControllerRef.current = null;
    }
  }

  // 发送图片到AI
  async function sendImageToAI(base64) {
    await sendToAI({ imageBase64: base64 });
  }

  // Run the self-test bank with the configured model.
  async function runSelfTest() {
    selfTestAbortRef.current?.abort();
    const controller = new AbortController();
    selfTestAbortRef.current = controller;
    try {
      const bank = await (await fetch(SELFTEST_PATH)).json();
      const state = {
        running: true,
        done: 0,
        total: bank.questions.length,
        correct: 0,
        misses: [],
      };
      setSelfTest({ ...state });
      for (const q of bank.questions) {
        if (controller.signal.aborted) break;
        let parsed = null;
        let failure = null;
        try {
          const response = await requestAI(
            aiConfig,
            { question: examPrompt({ text: selfTestText(q) }) },
            {
              mockTag: "ai-solver",
              jsonMode: true,
              maxTokens: 800,
              systemPrompt: examSystemPrompt(true),
              signal: controller.signal,
            }
          );
          parsed = parseExamAnswer(response.text);
        } catch (error) {
          if (error?.name === "AbortError") break;
          failure = getSafeAiErrorMessage(error);
        }
        if (parsed && matchesKey(parsed.answer, q.answer)) {
          state.correct += 1;
        } else {
          state.misses.push({
            id: q.id,
            question: q.question,
            key: q.answer,
            got: parsed ? parsed.answer : null,
            basis: q.basis,
            error: failure,
          });
        }
        state.done += 1;
        setSelfTest({ ...state, misses: [...state.misses] });
      }
      setSelfTest({ ...state, running: false, misses: [...state.misses] });
    } catch (error) {
      setSelfTest({
        running: false,
        done: 0,
        total: 0,
        correct: 0,
        misses: [],
        failed: error?.message || String(error),
      });
    } finally {
      if (selfTestAbortRef.current === controller)
        selfTestAbortRef.current = null;
    }
  }

  // Send text to AI
  async function sendTextToAI(text) {
    // Check if it's a special command
    if (text.startsWith("/preset ")) {
      const presetName = text.substring(8).trim().toLowerCase();
      const preset = promptPresets.find(
        (p) =>
          p.id.toLowerCase() === presetName ||
          p.name.toLowerCase() === presetName
      );

      if (preset) {
        choosePreset(preset.id);
        setRespText(
          tx(
            "Switched to preset: {0}\n\nDescription: {1}",
            preset.name,
            preset.description
          )
        );
        return;
      } else {
        setRespText(
          tx(
            'Preset "{0}" not found. Available presets: {1}',
            presetName,
            promptPresets.map((p) => p.name).join(", ")
          )
        );
        return;
      }
    }

    await sendToAI({ text });
  }

  // 处理文本提问
  async function handleTextQuestion() {
    if (!textInput.trim()) {
      setRespText("Please enter your question");
      return;
    }

    setRespText("");
    setBusy(true);
    try {
      await sendTextToAI(textInput.trim());
    } catch (e) {
      setRespText("Error: " + (e?.message || String(e)));
    } finally {
      setBusy(false);
    }
  }

  // Screen sharing stays open between questions: share once, then each
  // "Capture & ask" grabs the same area of the shared screen.
  function stopSharing() {
    shareRef.current?.stream.getTracks().forEach((track) => track.stop());
    shareRef.current = null;
    setSharing(false);
  }

  async function ensureShare() {
    if (shareRef.current) return shareRef.current;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
      throw new Error(tx("The browser does not support screen capture."));
    }
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: true,
      audio: false,
    });
    const video = document.createElement("video");
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    if (video.readyState < 1) {
      await new Promise((resolve, reject) => {
        video.onloadedmetadata = resolve;
        video.onerror = () =>
          reject(new Error(tx("The shared screen could not be read.")));
      });
    }
    await video.play();
    stream.getVideoTracks()[0]?.addEventListener("ended", stopSharing);
    shareRef.current = { stream, video };
    setSharing(true);
    return shareRef.current;
  }

  function grabFrame({ video }) {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx || !canvas.width || !canvas.height) {
      throw new Error(
        tx("The shared screen did not provide a usable video frame.")
      );
    }
    ctx.drawImage(video, 0, 0);
    return { canvas, width: canvas.width, height: canvas.height };
  }

  async function cropToBase64(canvas, box) {
    const crop = document.createElement("canvas");
    crop.width = Math.round(box.width);
    crop.height = Math.round(box.height);
    const ctx = crop.getContext("2d");
    if (!ctx)
      throw new Error(tx("Canvas rendering is not available in this browser."));
    ctx.drawImage(
      canvas,
      box.x,
      box.y,
      box.width,
      box.height,
      0,
      0,
      crop.width,
      crop.height
    );
    // PNG keeps small exam text sharp; JPEG for large areas.
    const type =
      crop.width * crop.height > 1600 * 1200 ? "image/jpeg" : "image/png";
    const blob = await new Promise((resolve) =>
      crop.toBlob((b) => resolve(b), type, 0.9)
    );
    if (!blob)
      throw new Error(tx("The browser could not encode the selected area."));
    setLastSizeKB(Math.round(blob.size / 1024));
    const dataUrl = await new Promise((res) => {
      const fr = new FileReader();
      fr.onload = () => res(fr.result);
      fr.readAsDataURL(blob);
    });
    return String(dataUrl);
  }

  // Capture the shared screen and choose (or adjust) the area.
  async function handleScreenshot() {
    setRespText("");
    setBusy(true);
    try {
      const frame = grabFrame(await ensureShare());
      setScreenshotData({
        ...frame,
        dataUrl: frame.canvas.toDataURL("image/jpeg", 0.85),
      });
      const last = lastBoxRef.current;
      if (
        last &&
        last.x + last.width <= frame.width &&
        last.y + last.height <= frame.height
      ) {
        setSelectionBox(last);
      } else {
        const size = Math.min(frame.width, frame.height) * 0.5;
        setSelectionBox({
          x: (frame.width - size) / 2,
          y: (frame.height - size) / 2,
          width: size,
          height: size,
        });
      }
      setIsSelecting(true);
    } catch (e) {
      if (e.name === "NotAllowedError") {
        setRespText(tx("Error: screen sharing was not allowed."));
      } else {
        setRespText(tx("Error: {0}", e?.message || String(e)));
      }
    } finally {
      setBusy(false);
    }
  }

  // One click per question: the same area of the shared screen.
  async function handleQuickCapture() {
    const box = lastBoxRef.current;
    if (!box) return handleScreenshot();
    setRespText("");
    setExamResult(null);
    setBusy(true);
    try {
      const frame = grabFrame(await ensureShare());
      const dataUrl = await cropToBase64(frame.canvas, box);
      await sendImageToAI(dataUrl);
    } catch (e) {
      setRespText(tx("Error: {0}", e?.message || String(e)));
    } finally {
      setBusy(false);
    }
  }

  // 确认选择区域并发送给AI
  async function handleConfirmSelection() {
    if (!screenshotData || !selectionBox) return;
    setBusy(true);
    setExamResult(null);
    try {
      lastBoxRef.current = selectionBox;
      setHasArea(true);
      const dataUrl = await cropToBase64(screenshotData.canvas, selectionBox);
      setScreenshotData(null);
      setSelectionBox(null);
      setIsSelecting(false);
      await sendImageToAI(dataUrl);
    } catch (e) {
      setRespText(tx("Error: {0}", e?.message || String(e)));
    } finally {
      setBusy(false);
    }
  }

  // 取消选择
  function handleCancelSelection() {
    setScreenshotData(null);
    setSelectionBox(null);
    setIsSelecting(false);
  }

  // 处理选择框拖拽 - 优化为自由框选
  function handleSelectionDrag(e, type, corner = null) {
    if (!screenshotData || !selectionBox) return;

    // 阻止默认行为和冒泡，防止干扰
    e.preventDefault();
    e.stopPropagation();

    // 获取图像容器的位置和尺寸信息
    const containerRect =
      screenshotViewportRef.current?.getBoundingClientRect();
    if (!containerRect?.width || !containerRect?.height) return;

    // 计算图像的实际尺寸与显示尺寸的比例
    const scaleX = screenshotData.width / containerRect.width;
    const scaleY = screenshotData.height / containerRect.height;

    // 保存初始状态，避免在拖动过程中使用可能变化的状态
    const initialBox = JSON.parse(JSON.stringify(selectionBox));
    const startX = e.clientX;
    const startY = e.clientY;

    const pointerId = e.pointerId;

    const handlePointerMove = (moveEvent) => {
      if (moveEvent.pointerId !== pointerId) return;
      // 阻止默认行为，防止选择文本等
      moveEvent.preventDefault();

      // 计算鼠标移动的像素距离
      const deltaPixelX = moveEvent.clientX - startX;
      const deltaPixelY = moveEvent.clientY - startY;

      // 将像素距离转换为图像坐标系中的距离
      const deltaX = deltaPixelX * scaleX;
      const deltaY = deltaPixelY * scaleY;

      // 根据操作类型计算新的选择框
      let newBox;

      if (type === "move") {
        // 移动操作 - 更新位置但保持大小不变
        const newX = Math.max(
          0,
          Math.min(
            screenshotData.width - initialBox.width,
            initialBox.x + deltaX
          )
        );
        const newY = Math.max(
          0,
          Math.min(
            screenshotData.height - initialBox.height,
            initialBox.y + deltaY
          )
        );
        newBox = {
          ...initialBox,
          x: newX,
          y: newY,
        };
      } else if (type === "resize" && corner) {
        const minimum = Math.min(
          50,
          screenshotData.width,
          screenshotData.height
        );
        let left = initialBox.x;
        let top = initialBox.y;
        let right = initialBox.x + initialBox.width;
        let bottom = initialBox.y + initialBox.height;

        if (corner.includes("left"))
          left = Math.max(0, Math.min(right - minimum, left + deltaX));
        if (corner.includes("right"))
          right = Math.min(
            screenshotData.width,
            Math.max(left + minimum, right + deltaX)
          );
        if (corner.includes("top"))
          top = Math.max(0, Math.min(bottom - minimum, top + deltaY));
        if (corner.includes("bottom"))
          bottom = Math.min(
            screenshotData.height,
            Math.max(top + minimum, bottom + deltaY)
          );

        newBox = { x: left, y: top, width: right - left, height: bottom - top };
      }

      if (newBox) setSelectionBox(newBox);
    };

    const cleanup = () => {
      document.removeEventListener("pointermove", handlePointerMove);
      document.removeEventListener("pointerup", handlePointerUp);
      document.removeEventListener("pointercancel", handlePointerUp);
      dragCleanupRef.current = null;
    };
    const handlePointerUp = (upEvent) => {
      if (upEvent.pointerId === pointerId) cleanup();
    };

    dragCleanupRef.current?.();
    dragCleanupRef.current = cleanup;
    document.addEventListener("pointermove", handlePointerMove, {
      passive: false,
    });
    document.addEventListener("pointerup", handlePointerUp);
    document.addEventListener("pointercancel", handlePointerUp);
  }

  // 拍照功能
  async function handleShoot() {
    if (!videoRef.current) return;
    setRespText("");
    setBusy(true);
    try {
      const blob = await captureCompressedJpeg(videoRef.current);
      setLastSizeKB(Math.round(blob.size / 1024));

      const dataUrl = await new Promise((res) => {
        const fr = new FileReader();
        fr.onload = () => res(fr.result);
        fr.readAsDataURL(blob);
      });
      const base64 = String(dataUrl).split(",")[1];

      await sendImageToAI(base64);
    } catch (e) {
      setRespText("Error: " + (e?.message || String(e)));
    } finally {
      setBusy(false);
    }
  }

  return (
    <AppScaffold appId="solver">
      <div className={`app-container ${styles.solver}`}>
        <p className={styles.lead}>
          {tx(
            "Ask with your camera, a screen capture, or plain text. Use the private local demo, or connect your own AI provider for this page session."
          )}
        </p>

        <div className={styles.grid}>
          <section
            className={styles.panel}
            aria-labelledby="solver-input-title"
          >
            <div className={styles.panelHead}>
              <Heading
                as="h2"
                id="solver-input-title"
                className={styles.panelTitle}
              >
                {tx("Input")}
              </Heading>
              <fieldset className={styles.modeGroup}>
                <legend className={styles.srOnly}>{tx("Input mode")}</legend>
                <div className={styles.segmented}>
                  {INPUT_MODES.map((mode) => (
                    <label
                      key={mode.id}
                      className={
                        captureMode === mode.id
                          ? styles.segmentActive
                          : styles.segment
                      }
                    >
                      <input
                        type="radio"
                        name="solver-input-mode"
                        value={mode.id}
                        checked={captureMode === mode.id}
                        onChange={(e) => setCaptureMode(e.target.value)}
                      />
                      <span>{mode.label}</span>
                    </label>
                  ))}
                </div>
              </fieldset>
            </div>

            {/* Camera Preview */}
            {captureMode === "camera" && (
              <div>
                <div
                  className={styles.stage}
                  data-live={ready ? "true" : "false"}
                >
                  <video
                    ref={videoRef}
                    autoPlay
                    muted
                    playsInline
                    className={styles.video}
                  />
                  {!ready && (
                    <span className={styles.stageHint}>
                      {starting
                        ? tx("Waiting for camera permission…")
                        : tx("Camera off")}
                    </span>
                  )}
                </div>
                <div role="status" className={styles.note}>
                  {ready
                    ? tx("Camera ready — video stays on this device.")
                    : starting
                    ? tx("Waiting for camera permission…")
                    : error
                    ? tx("Camera error: {0}", error)
                    : tx(
                        "Camera is off. It starts only after you choose Enable camera."
                      )}
                </div>
                <div className={styles.actions}>
                  {!ready ? (
                    <button
                      type="button"
                      className="button button--primary"
                      onClick={startCamera}
                      disabled={starting}
                    >
                      {starting ? tx("Starting…") : tx("Enable camera")}
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="button button--secondary"
                      onClick={stopCamera}
                    >
                      {tx("Turn off camera")}
                    </button>
                  )}
                  <button
                    type="button"
                    className="button button--primary"
                    onClick={handleShoot}
                    disabled={!ready || busy}
                  >
                    {busy ? tx("Processing…") : tx("Capture and solve")}
                  </button>
                </div>
              </div>
            )}

            {/* Screenshot Mode */}
            {captureMode === "screenshot" && (
              <div>
                {!isSelecting ? (
                  // 初始状态 - 显示截图按钮
                  <>
                    <div className={styles.stage}>
                      <span className={styles.stageHint}>
                        {sharing && hasArea
                          ? tx("Screen shared · area set")
                          : tx("No screen captured yet")}
                      </span>
                    </div>
                    <div className={styles.note}>
                      {sharing && hasArea
                        ? tx(
                            "Sharing stays on: show the next question in the shared window, then press Capture & ask. The same area is captured each time."
                          )
                        : tx(
                            "Share the window with the questions once, drag the box over the question area, and analyze. After that, each question is one click."
                          )}
                    </div>
                    <div className={styles.actions}>
                      {sharing && hasArea ? (
                        <>
                          <button
                            type="button"
                            className="button button--primary"
                            onClick={handleQuickCapture}
                            disabled={busy}
                          >
                            {busy ? tx("Processing…") : tx("Capture & ask")}
                          </button>
                          <button
                            type="button"
                            className="button button--secondary"
                            onClick={handleScreenshot}
                            disabled={busy}
                          >
                            {tx("Adjust area")}
                          </button>
                          <button
                            type="button"
                            className="button button--secondary"
                            onClick={stopSharing}
                            disabled={busy}
                          >
                            {tx("Stop sharing")}
                          </button>
                        </>
                      ) : (
                        <button
                          type="button"
                          className="button button--primary"
                          onClick={handleScreenshot}
                          disabled={busy}
                        >
                          {busy ? tx("Processing…") : tx("Capture screen")}
                        </button>
                      )}
                    </div>
                  </>
                ) : (
                  // 选择状态 - 显示截图和选择框
                  <>
                    <div
                      ref={screenshotViewportRef}
                      style={{
                        position: "relative",
                        width: "min(560px, 100%)",
                        aspectRatio: screenshotData
                          ? `${screenshotData.width} / ${screenshotData.height}`
                          : "4 / 3",
                        border: "1px solid var(--ds-line)",
                        borderRadius: 18,
                        overflow: "hidden",
                        touchAction: "none",
                        background: "#000",
                      }}
                    >
                      <img
                        src={screenshotData?.dataUrl}
                        alt={tx("Screenshot")}
                        draggable={false}
                        style={{
                          width: "100%",
                          height: "100%",
                          objectFit: "fill",
                          userSelect: "none",
                        }}
                        onPointerDown={(e) => handleSelectionDrag(e, "move")}
                      />
                      {selectionBox && screenshotData && (
                        <div
                          style={{
                            position: "absolute",
                            left: `${
                              (selectionBox.x / screenshotData.width) * 100
                            }%`,
                            top: `${
                              (selectionBox.y / screenshotData.height) * 100
                            }%`,
                            width: `${
                              (selectionBox.width / screenshotData.width) * 100
                            }%`,
                            height: `${
                              (selectionBox.height / screenshotData.height) *
                              100
                            }%`,
                            border: "2px solid var(--ifm-color-emphasis-800)",
                            backgroundColor: "rgba(0, 0, 0, 0.06)",
                            cursor: "move",
                            boxSizing: "border-box",
                          }}
                          onPointerDown={(e) => {
                            e.stopPropagation();
                            handleSelectionDrag(e, "move");
                          }}
                        >
                          {/* 八个方向的调整大小手柄 */}
                          {/* 左上角 */}
                          <div
                            style={{
                              position: "absolute",
                              left: -8,
                              top: -8,
                              width: 16,
                              height: 16,
                              backgroundColor: "var(--ifm-color-emphasis-800)",
                              cursor: "nw-resize",
                              borderRadius: "50%",
                            }}
                            onPointerDown={(e) => {
                              e.stopPropagation();
                              handleSelectionDrag(e, "resize", "top-left");
                            }}
                          />
                          {/* 上边 */}
                          <div
                            style={{
                              position: "absolute",
                              left: "50%",
                              top: -8,
                              width: 16,
                              height: 16,
                              backgroundColor: "var(--ifm-color-emphasis-800)",
                              cursor: "n-resize",
                              borderRadius: "50%",
                              transform: "translateX(-50%)",
                            }}
                            onPointerDown={(e) => {
                              e.stopPropagation();
                              handleSelectionDrag(e, "resize", "top");
                            }}
                          />
                          {/* 右上角 */}
                          <div
                            style={{
                              position: "absolute",
                              right: -8,
                              top: -8,
                              width: 16,
                              height: 16,
                              backgroundColor: "var(--ifm-color-emphasis-800)",
                              cursor: "ne-resize",
                              borderRadius: "50%",
                            }}
                            onPointerDown={(e) => {
                              e.stopPropagation();
                              handleSelectionDrag(e, "resize", "top-right");
                            }}
                          />
                          {/* 右边 */}
                          <div
                            style={{
                              position: "absolute",
                              right: -8,
                              top: "50%",
                              width: 16,
                              height: 16,
                              backgroundColor: "var(--ifm-color-emphasis-800)",
                              cursor: "e-resize",
                              borderRadius: "50%",
                              transform: "translateY(-50%)",
                            }}
                            onPointerDown={(e) => {
                              e.stopPropagation();
                              handleSelectionDrag(e, "resize", "right");
                            }}
                          />
                          {/* 右下角 */}
                          <div
                            style={{
                              position: "absolute",
                              right: -8,
                              bottom: -8,
                              width: 16,
                              height: 16,
                              backgroundColor: "var(--ifm-color-emphasis-800)",
                              cursor: "se-resize",
                              borderRadius: "50%",
                            }}
                            onPointerDown={(e) => {
                              e.stopPropagation();
                              handleSelectionDrag(e, "resize", "bottom-right");
                            }}
                          />
                          {/* 下边 */}
                          <div
                            style={{
                              position: "absolute",
                              left: "50%",
                              bottom: -8,
                              width: 16,
                              height: 16,
                              backgroundColor: "var(--ifm-color-emphasis-800)",
                              cursor: "s-resize",
                              borderRadius: "50%",
                              transform: "translateX(-50%)",
                            }}
                            onPointerDown={(e) => {
                              e.stopPropagation();
                              handleSelectionDrag(e, "resize", "bottom");
                            }}
                          />
                          {/* 左下角 */}
                          <div
                            style={{
                              position: "absolute",
                              left: -8,
                              bottom: -8,
                              width: 16,
                              height: 16,
                              backgroundColor: "var(--ifm-color-emphasis-800)",
                              cursor: "sw-resize",
                              borderRadius: "50%",
                            }}
                            onPointerDown={(e) => {
                              e.stopPropagation();
                              handleSelectionDrag(e, "resize", "bottom-left");
                            }}
                          />
                          {/* 左边 */}
                          <div
                            style={{
                              position: "absolute",
                              left: -8,
                              top: "50%",
                              width: 16,
                              height: 16,
                              backgroundColor: "var(--ifm-color-emphasis-800)",
                              cursor: "w-resize",
                              borderRadius: "50%",
                              transform: "translateY(-50%)",
                            }}
                            onPointerDown={(e) => {
                              e.stopPropagation();
                              handleSelectionDrag(e, "resize", "left");
                            }}
                          />
                        </div>
                      )}
                    </div>
                    <div className={styles.note}>
                      {tx(
                        "Drag the box to move it, or drag a corner to resize the selection."
                      )}
                    </div>
                    <div className={styles.actions}>
                      <button
                        type="button"
                        className="button button--primary"
                        onClick={handleConfirmSelection}
                        disabled={busy}
                      >
                        {busy ? tx("Processing…") : tx("Analyze selection")}
                      </button>
                      <button
                        type="button"
                        className="button button--secondary"
                        onClick={handleCancelSelection}
                        disabled={busy}
                      >
                        {tx("Cancel")}
                      </button>
                    </div>
                  </>
                )}
              </div>
            )}

            {/* Text Question Mode */}
            {captureMode === "text" && (
              <div>
                <label
                  htmlFor="solver-text-question"
                  className={styles.fieldLabel}
                >
                  {tx("Your question")}
                </label>
                <textarea
                  id="solver-text-question"
                  className={styles.question}
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder={tx(
                    "For example:\n• Explain the basic principles of quantum mechanics\n• Write a Python sorting algorithm\n• Type /preset to see available presets"
                  )}
                />
                <div className={styles.note}>
                  {tx("Tip: type ")}
                  <code>/preset {tx("name")}</code>
                  {tx(" to switch presets, e.g.")}{" "}
                  <code>/preset {tx("Math Problem Solver")}</code>.
                </div>
                <div className={styles.actions}>
                  <button
                    type="button"
                    className="button button--primary"
                    onClick={handleTextQuestion}
                    disabled={busy || !textInput.trim()}
                  >
                    {busy ? tx("Thinking…") : tx("Ask")}
                  </button>
                </div>
              </div>
            )}

            {examResult?.main && (
              <p
                className={
                  examResult.agree === false || examResult.checkFailed
                    ? styles.quickAnswerWarn
                    : styles.quickAnswer
                }
                role="status"
              >
                {tx("Answer:")}{" "}
                <strong>
                  {answerLabel(examResult.main.answer, {
                    yes: tx("True"),
                    no: tx("False"),
                  })}
                </strong>
                {" · "}
                {examResult.agree === true
                  ? tx("both answers agree")
                  : examResult.agree === false
                  ? tx("answers differ — check it yourself")
                  : examResult.checkFailed
                  ? tx("not checked")
                  : tx("single answer")}
              </p>
            )}

            {lastSizeKB != null && (
              <div className={styles.meta}>
                {tx("Last image size: ")}
                {lastSizeKB} KB
              </div>
            )}
          </section>

          <section
            className={styles.panel}
            aria-labelledby="solver-model-title"
          >
            <div className={styles.panelHead}>
              <Heading
                as="h2"
                id="solver-model-title"
                className={styles.panelTitle}
              >
                {tx("Model & prompt")}
              </Heading>
            </div>
            <AIProviderSettings
              value={aiConfig}
              onChange={setAiConfig}
              title={tx("Solver model")}
              requireVision={captureMode !== "text"}
            />

            <div className={styles.promptBlock}>
              <div>
                <label className={styles.fieldLabel}>
                  {tx("Prompt preset")}
                  <select
                    value={selectedPreset}
                    onChange={(e) => choosePreset(e.target.value)}
                    className={styles.select}
                  >
                    {promptPresets.map((preset) => (
                      <option key={preset.id} value={preset.id}>
                        {preset.name}
                      </option>
                    ))}
                  </select>
                </label>
                <div className={styles.note}>
                  {promptPresets.find((p) => p.id === selectedPreset)
                    ?.description || ""}
                </div>
              </div>

              {isExam && (
                <div className={styles.examTools}>
                  <label className={styles.checkRow}>
                    <input
                      type="checkbox"
                      checked={doubleCheck}
                      onChange={(e) => setDoubleCheck(e.target.checked)}
                    />
                    <span>
                      {tx(
                        "Check each answer with a second independent answer (two requests per question)"
                      )}
                    </span>
                  </label>
                  <div className={styles.selfTest}>
                    <strong>{tx("Self-test")}</strong>
                    <p className={styles.note}>
                      {tx(
                        "Before an exam, measure how well the selected model answers 32 lab safety questions with settled answers."
                      )}
                    </p>
                    <div className={styles.actions}>
                      {selfTest?.running ? (
                        <button
                          type="button"
                          className="button button--secondary"
                          onClick={() => selfTestAbortRef.current?.abort()}
                        >
                          {tx("Stop ({0}/{1})", selfTest.done, selfTest.total)}
                        </button>
                      ) : (
                        <button
                          type="button"
                          className="button button--secondary"
                          onClick={runSelfTest}
                          disabled={!usesModel || busy}
                          title={
                            usesModel
                              ? undefined
                              : tx("Choose a model provider first.")
                          }
                        >
                          {tx("Run self-test")}
                        </button>
                      )}
                    </div>
                    {selfTest && !selfTest.failed && selfTest.done > 0 && (
                      <div className={styles.selfTestResult} role="status">
                        <p>
                          {tx(
                            "{0} of {1} correct ({2}%)",
                            selfTest.correct,
                            selfTest.done,
                            Math.round((100 * selfTest.correct) / selfTest.done)
                          )}
                          {selfTest.running ? " …" : ""}
                        </p>
                        {selfTest.misses.length > 0 && (
                          <details>
                            <summary>
                              {tx(
                                "Questions answered wrongly ({0})",
                                selfTest.misses.length
                              )}
                            </summary>
                            <ol>
                              {selfTest.misses.map((m) => (
                                <li key={m.id}>
                                  {m.question}{" "}
                                  <span>
                                    {tx(
                                      "Key: {0}",
                                      answerLabel(
                                        typeof m.key === "boolean"
                                          ? { kind: "bool", value: m.key }
                                          : { kind: "letters", letters: m.key },
                                        { yes: tx("True"), no: tx("False") }
                                      )
                                    )}
                                    {" · "}
                                    {m.error
                                      ? m.error
                                      : tx(
                                          "Model: {0}",
                                          m.got
                                            ? answerLabel(m.got, {
                                                yes: tx("True"),
                                                no: tx("False"),
                                              })
                                            : tx("no answer")
                                        )}
                                  </span>
                                </li>
                              ))}
                            </ol>
                          </details>
                        )}
                      </div>
                    )}
                    {selfTest?.failed && (
                      <p className={styles.note}>
                        {tx("Self-test failed: {0}", selfTest.failed)}
                      </p>
                    )}
                  </div>
                </div>
              )}

              {selectedPreset === "custom" ? (
                <label className={styles.fieldLabel}>
                  {tx("Question for image mode")}
                  <textarea
                    className={styles.questionSmall}
                    value={question}
                    onChange={(e) => {
                      setQuestion(e.target.value);
                    }}
                    rows={3}
                  />
                </label>
              ) : (
                <div className={styles.note}>
                  {tx(
                    "Preset instructions are applied automatically for the selected mode."
                  )}
                </div>
              )}
            </div>
          </section>
        </div>

        <section
          className={`${styles.panel} ${styles.responsePanel}`}
          aria-labelledby="solver-response-title"
        >
          <div className={styles.panelHead}>
            <Heading
              as="h2"
              id="solver-response-title"
              className={styles.panelTitle}
            >
              {tx("Response")}
            </Heading>
          </div>
          {examResult && (
            <div className={styles.answerCard} role="status">
              {examResult.main ? (
                <>
                  <div className={styles.answerTop}>
                    <span className={styles.answerValue}>
                      {answerLabel(examResult.main.answer, {
                        yes: tx("True"),
                        no: tx("False"),
                      })}
                    </span>
                    <span className={styles.answerMeta}>
                      {[
                        TYPE_LABELS[examResult.main.type],
                        CONFIDENCE_LABELS[examResult.main.confidence],
                      ]
                        .filter(Boolean)
                        .join(" · ")}
                    </span>
                  </div>
                  {examResult.agree === true && (
                    <p className={styles.agree}>
                      {tx("A second, independent answer agrees.")}
                    </p>
                  )}
                  {examResult.agree === false && (
                    <p className={styles.disagree}>
                      {tx(
                        "The second answer differs ({0}). Read the question again and decide yourself.",
                        answerLabel(examResult.check.answer, {
                          yes: tx("True"),
                          no: tx("False"),
                        })
                      )}
                    </p>
                  )}
                  {examResult.checkFailed && (
                    <p className={styles.disagree}>
                      {tx("The second answer could not be obtained.")}
                    </p>
                  )}
                  {examResult.main.question && (
                    <p className={styles.readQuestion}>
                      <strong>{tx("Question as read:")}</strong>{" "}
                      {examResult.main.question}
                    </p>
                  )}
                  {examResult.main.reason && <p>{examResult.main.reason}</p>}
                  {examResult.main.basis && (
                    <p className={styles.note}>
                      {tx("Basis:")} {examResult.main.basis}
                    </p>
                  )}
                </>
              ) : (
                <p className={styles.disagree}>
                  {tx(
                    "No answer could be read from the model's reply; see the full reply below."
                  )}
                </p>
              )}
            </div>
          )}
          {examResult ? (
            <details className={styles.rawReply}>
              <summary>{tx("Full model reply")}</summary>
              <div className={styles.response}>{respText}</div>
            </details>
          ) : (
            <div
              role="status"
              aria-live="polite"
              aria-busy={busy}
              tabIndex={0}
              className={styles.response}
            >
              {respText || (
                <span className={styles.empty}>
                  {busy
                    ? tx("Waiting for the model…")
                    : tx("The answer appears here after you capture or ask.")}
                </span>
              )}
            </div>
          )}
        </section>
        <CitationNotice />
      </div>
    </AppScaffold>
  );
}
