
import { useEffect, useMemo, useRef, useState } from 'react';

const HARDCODED_API_ENDPOINT = 'https://api.hunyuan.cloud.tencent.com/v1/chat/completions';
const HARDCODED_API_KEY = 'sk-JdwAvFcfyW5ngP2i3cpeB43QrR92gjnRcNzKkMfpcEVu8hlE';

function computeDefaultApiEndpoint() {
  return HARDCODED_API_ENDPOINT;
}

function useCamera() {
  const videoRef = useRef(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let stream = null;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment' },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
          setReady(true);
        }
      } catch (e) {
        setError(e?.message || String(e));
      }
    })();

    return () => {
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  return { videoRef, ready, error };
}

async function captureCompressedJpeg(video, maxSide = 1280, quality = 0.85) {
  const w = video.videoWidth || 1080;
  const h = video.videoHeight || 1440;
  const scale = Math.min(1, maxSide / Math.max(w, h));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(w * scale);
  canvas.height = Math.round(h * scale);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return await new Promise((resolve) => canvas.toBlob((b) => resolve(b), 'image/jpeg', quality));
}

async function postJson(url, json, extraHeaders = {}) {
  // Mock mode: allow using a fake upstream when url starts with "mock://"
  if (typeof url === 'string' && url.startsWith('mock://')) {
    const now = new Date().toISOString();
    const mockText = (() => {
      const q = json?.question || 'No question provided';
      const model = json?.model || 'hunyuan-lite';
      const hasImage = !!json?.imageBase64 || !!json?.imageUrl;
      const header = hasImage ? 'Mock Vision Analysis' : 'Mock Text Analysis';
      return `${header} (model: ${model})\n\nUser question:\n${q}\n\nThis is a mocked response for demo purposes. Replace proxy URL with your real API when ready.`;
    })();

    const body = {
      Response: {
        RequestId: 'mock-' + Math.random().toString(36).slice(2),
        Choices: [
          {
            Message: {
              Content: mockText,
            },
          },
        ],
        Usage: {
          PromptTokens: 128,
          CompletionTokens: 256,
          TotalTokens: 384,
        },
        Timestamp: now,
      },
    };

    return {
      ok: true,
      status: 200,
      json: async () => body,
      text: async () => JSON.stringify(body),
      headers: new Map([['content-type', 'application/json']]),
    };
  }

  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...extraHeaders },
    body: JSON.stringify(json),
  });
}

function formatAiResponse(data) {
  if (Array.isArray(data?.choices) && data.choices.length > 0) {
    const aiMessage = data.choices[0].message?.content || '';
    const usage = data.usage || {};
    const formattedResponse = `${aiMessage}\n\n---\nTokens: ${usage.total_tokens || 'N/A'} (Prompt: ${usage.prompt_tokens || 'N/A'}, Completion: ${usage.completion_tokens || 'N/A'})`;
    return { text: formattedResponse, raw: data };
  }

  if (data?.Response && Array.isArray(data.Response.Choices) && data.Response.Choices.length > 0) {
    const legacyMessage = data.Response.Choices[0].Message?.Content || '';
    const legacyUsage = data.Response.Usage || {};
    const formattedLegacy = `${legacyMessage}\n\n---\nTokens: ${legacyUsage.TotalTokens || 'N/A'} (Prompt: ${legacyUsage.PromptTokens || 'N/A'}, Completion: ${legacyUsage.CompletionTokens || 'N/A'})`;
    return { text: formattedLegacy, raw: data };
  }

  const fallback = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
  return { text: fallback, raw: data };
}

function normalizeBase64Image(imageBase64) {
  if (!imageBase64) return null;
  if (imageBase64.startsWith('data:')) {
    return imageBase64;
  }
  return `data:image/jpeg;base64,${imageBase64}`;
}

function buildHunyuanPayload({ question, imageBase64, imageUrl, model }) {
  const contents = [];
  const trimmedQuestion = (question || '').trim();

  if (trimmedQuestion) {
    contents.push({
      type: 'text',
      text: trimmedQuestion,
    });
  }

  const normalizedImageUrl = (imageUrl || '').trim();
  const normalizedBase64 = normalizeBase64Image(imageBase64);

  if (normalizedImageUrl) {
    contents.push({
      type: 'image_url',
      image_url: { url: normalizedImageUrl },
    });
  } else if (normalizedBase64) {
    contents.push({
      type: 'image_url',
      image_url: { url: normalizedBase64 },
    });
  }

  if (contents.length === 0) {
    contents.push({
      type: 'text',
      text: '你好',
    });
  }

  return {
    model: model || 'hunyuan-vision',
    messages: [
      {
        role: 'user',
        content: contents,
      },
    ],
    stream: false,
  };
}

export default function SolverAppPage() {
  const defaultApiEndpoint = useMemo(computeDefaultApiEndpoint, []);
  const { videoRef, ready, error } = useCamera();

  // State management
  const [apiUrl, setApiUrl] = useState('');
  const [useDefaultApi, setUseDefaultApi] = useState(false);
  // Preset prompt list
// Preset prompt list — full English version (ready to use with HunYuan API)
const promptPresets = [
  { 
    id: 'default', 
    name: 'Intelligent Analysis Assistant', 
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
    description: 'General intelligent analysis and solution generation'
  },

  { 
    id: 'math', 
    name: 'Math Problem Solver', 
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
    description: 'Equation solving, geometry, algebra, and other math problems'
  },

  { 
    id: 'lab_safety', 
    name: 'Laboratory Safety Assessment', 
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
    description: 'Laboratory safety risk assessment and improvement suggestions'
  },

  { 
    id: 'plant', 
    name: 'Plant Identification Analysis', 
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
    description: 'Plant species identification and growth information'
  },

  { 
    id: 'code', 
    name: 'Code Review and Optimization', 
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
    description: 'Code explanation, debugging, and performance optimization'
  },

  { 
    id: 'translation', 
    name: 'Academic Literature Translation', 
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
    description: 'Professional academic translation and terminology explanation'
  },

  { 
    id: 'physics', 
    name: 'Physics Problem Solver', 
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
    description: 'Mechanics, electromagnetism, thermodynamics, and other physics problems'
  },

  { 
    id: 'chemistry', 
    name: 'Chemistry Problem Solver', 
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
    description: 'Chemical equations, structures, mechanisms, and experiment analysis'
  },

  { 
    id: 'english', 
    name: 'English Learning Assistant', 
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
    description: 'Grammar explanation, vocabulary learning, and exercise guidance'
  },

  { 
    id: 'ocr', 
    name: 'Text Extraction', 
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
    description: 'Accurate image text recognition and format preservation'
  },

  { 
    id: 'biology', 
    name: 'Biology Analysis', 
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
    description: 'Biological structures, cell biology, and ecology analysis'
  },

  { 
    id: 'history', 
    name: 'Historical Document Interpretation', 
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
    description: 'Historical document or artifact interpretation'
  },

  { 
    id: 'medical', 
    name: 'Medical Image Analysis', 
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
    description: 'Medical image structure and general interpretation (non-diagnostic)'
  },

  { 
    id: 'engineering', 
    name: 'Engineering Drawing Interpretation', 
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
    description: 'Engineering drawings, technical charts, and design interpretation'
  },

  { 
    id: 'finance', 
    name: 'Financial Statement Analysis', 
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
    description: 'Financial statement interpretation and business performance analysis'
  },

  { 
    id: 'art', 
    name: 'Artwork Appreciation', 
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
    description: 'Art style, technique, and aesthetic value analysis'
  },

  { 
    id: 'legal', 
    name: 'Legal Document Interpretation', 
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
    description: 'Legal document and contract clause interpretation (non-advisory)'
  },

  { 
    id: 'education', 
    name: 'Educational Material Analysis', 
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
    description: 'Educational material evaluation and teaching method design'
  },

  { 
    id: 'custom', 
    name: 'Custom Mode', 
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
    description: 'Flexible mode for custom or mixed scenarios'
  }
];


  const [selectedPreset, setSelectedPreset] = useState('default');
  const [question, setQuestion] = useState(promptPresets[0].prompt);
  const [model, setModel] = useState('hunyuan-vision');
  const [respText, setRespText] = useState('');
  const [busy, setBusy] = useState(false);
  const [lastSizeKB, setLastSizeKB] = useState(null);
  const [captureMode, setCaptureMode] = useState('camera'); // 'camera', 'screenshot', 'text'
  const [textInput, setTextInput] = useState(''); // Text input
  const [screenshotData, setScreenshotData] = useState(null); // Screenshot data
  const [selectionBox, setSelectionBox] = useState(null); // Selection box
  const [isSelecting, setIsSelecting] = useState(false); // Whether selecting

  // 通用的发送到AI的函数
  async function sendToAI(payload) {
    // Determine which API to use: default API, custom API, or mock
    let targetUrl;
    const attemptedDefault = !!useDefaultApi;
    if (useDefaultApi) {
      targetUrl = defaultApiEndpoint;
    } else if (apiUrl && apiUrl.trim() && /^https?:\/\//.test(apiUrl)) {
      targetUrl = apiUrl;
    } else {
      targetUrl = 'mock://ai-solver';
    }

    try {
      let requestBody = payload;
      let requestHeaders = {};
      const usingHardcodedDefault = targetUrl === HARDCODED_API_ENDPOINT;

      if (usingHardcodedDefault) {
        requestBody = buildHunyuanPayload(payload);
        requestHeaders = { Authorization: `Bearer ${HARDCODED_API_KEY}` };
      }

      const response = await postJson(targetUrl, requestBody, requestHeaders);

      if (!response.ok) {
        const rawText = await response.text().catch(() => '');
        if (attemptedDefault && (response.status === 404 || response.status === 405 || response.status === 403)) {
          const snippet = rawText ? rawText.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 120) : '';
          const note = `Default API returned ${response.status}${snippet ? ` (${snippet})` : ''}. Using mock response instead.`;
          const fallbackResp = await postJson('mock://ai-solver', payload);
          const fallbackData = await fallbackResp.json();
          const { text } = formatAiResponse(fallbackData);
          setRespText(`${note}\n\n${text}`);
          window.lastFullResponse = fallbackData;
          return;
        }

        if (response.status === 404) {
          const guidance = useDefaultApi
            ? 'The built-in solver endpoint (/api/solve) is not reachable. Deploy the serverless function or point the app to your own API in the settings.'
            : 'The URL you entered cannot be found. Double-check the API path or start your local proxy (node local-server.js).';
          throw new Error(`Request failed 404: ${guidance}`);
        }

        const snippet = rawText
          ? rawText.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 180)
          : '';
        const detail = snippet ? ` Response: ${snippet}` : '';
        throw new Error(`Request failed ${response.status}.${detail}`);
      }
      const data = await response.json().catch(async () => ({ raw: await response.text() }));
      const { text } = formatAiResponse(data);
      setRespText(text);
      window.lastFullResponse = data;
    } catch (e) {
      if (e.name === 'TypeError' && e.message.includes('fetch')) {
        throw new Error(`Network error: Cannot connect to ${targetUrl}. Please verify the API server or the URL in the settings.`);
      }
      throw e instanceof Error ? e : new Error(String(e));
    }
  }

  // 发送图片到AI
  async function sendImageToAI(base64) {
    const payload = { imageBase64: base64, question, model };
    await sendToAI(payload);
  }

  // Send text to AI
  async function sendTextToAI(text) {
    // Check if it's a special command
    if (text.startsWith('/preset ')) {
      const presetName = text.substring(8).trim().toLowerCase();
      const preset = promptPresets.find(p => p.id.toLowerCase() === presetName || p.name.toLowerCase() === presetName);
      
      if (preset) {
        setSelectedPreset(preset.id);
        if (preset.id === 'custom') {
          setQuestion('');
        } else {
          setQuestion(preset.prompt);
        }
        setRespText(`Switched to preset: ${preset.name}\n\nDescription: ${preset.description}`);
        return;
      } else {
        setRespText(`Preset "${presetName}" not found. Available presets: ${promptPresets.map(p => p.name).join(', ')}`);
        return;
      }
    }
    
    const payload = { question: text, model: 'hunyuan-lite' };
    await sendToAI(payload);
  }

  // 处理文本提问
  async function handleTextQuestion() {
    if (!textInput.trim()) {
      setRespText('Please enter your question');
      return;
    }

    setRespText('');
    setBusy(true);
    try {
      await sendTextToAI(textInput.trim());
    } catch (e) {
      setRespText('Error: ' + (e?.message || String(e)));
    } finally {
      setBusy(false);
    }
  }

  // 截图功能 - 第一步：捕获整个屏幕
  async function handleScreenshot() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
      setRespText('Error: The browser does not support screen capture');
      return;
    }

    setRespText('');
    setBusy(true);
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { mediaSource: 'screen' },
        audio: false
      });

      const video = document.createElement('video');
      video.srcObject = stream;
      video.play();

      await new Promise((resolve) => {
        video.onloadedmetadata = resolve;
      });

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);

      stream.getTracks().forEach(track => track.stop());

      const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
      
      // 保存截图数据，进入选择模式
      setScreenshotData({
        dataUrl,
        width: canvas.width,
        height: canvas.height,
        canvas
      });
      
      // 初始化选择框（默认选择中间区域）
      const defaultSize = Math.min(canvas.width, canvas.height) * 0.5;
      setSelectionBox({
        x: (canvas.width - defaultSize) / 2,
        y: (canvas.height - defaultSize) / 2,
        width: defaultSize,
        height: defaultSize
      });
      
      setIsSelecting(true);

    } catch (e) {
      if (e.name === 'NotAllowedError') {
        setRespText('Error: The user canceled screen sharing permission');
      } else {
        setRespText('Error: ' + (e?.message || String(e)));
      }
    } finally {
      setBusy(false);
    }
  }

  // 确认选择区域并发送给AI
  async function handleConfirmSelection() {
    if (!screenshotData || !selectionBox) return;
    
    setBusy(true);
    try {
      // 创建新的canvas来裁剪选中区域
      const cropCanvas = document.createElement('canvas');
      cropCanvas.width = selectionBox.width;
      cropCanvas.height = selectionBox.height;
      const cropCtx = cropCanvas.getContext('2d');
      
      // 从原始canvas裁剪选中区域
      cropCtx.drawImage(
        screenshotData.canvas,
        selectionBox.x, selectionBox.y, selectionBox.width, selectionBox.height,
        0, 0, selectionBox.width, selectionBox.height
      );
      
      const blob = await new Promise((resolve) => 
        cropCanvas.toBlob((b) => resolve(b), 'image/jpeg', 0.85)
      );
      
      setLastSizeKB(Math.round(blob.size / 1024));

      const dataUrl = await new Promise((res) => {
        const fr = new FileReader();
        fr.onload = () => res(fr.result);
        fr.readAsDataURL(blob);
      });
      const base64 = String(dataUrl).split(',')[1];

      await sendImageToAI(base64);
      
      // 清理状态
      setScreenshotData(null);
      setSelectionBox(null);
      setIsSelecting(false);

    } catch (e) {
      setRespText('Error: ' + (e?.message || String(e)));
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
    const containerRect = e.currentTarget.getBoundingClientRect();
    
    // 计算图像的实际尺寸与显示尺寸的比例
    const scaleX = screenshotData.width / containerRect.width;
    const scaleY = screenshotData.height / containerRect.height;
    
    // 保存初始状态，避免在拖动过程中使用可能变化的状态
    const initialBox = JSON.parse(JSON.stringify(selectionBox));
    const startX = e.clientX;
    const startY = e.clientY;
    
    // 创建一个引用，用于存储最新的选择框状态
    const currentBoxRef = { ...initialBox };
    
    const handleMouseMove = (moveEvent) => {
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
      
      if (type === 'move') {
        // 移动操作 - 更新位置但保持大小不变
        const newX = Math.max(0, Math.min(screenshotData.width - initialBox.width, initialBox.x + deltaX));
        const newY = Math.max(0, Math.min(screenshotData.height - initialBox.height, initialBox.y + deltaY));
        newBox = { 
          ...initialBox, 
          x: newX, 
          y: newY 
        };
      } else if (type === 'resize' && corner) {
        // 自由调整大小操作 - 根据拖拽的角落调整
        switch (corner) {
          case 'top-left':
            newBox = {
              x: Math.max(0, Math.min(initialBox.x + initialBox.width - 50, initialBox.x + deltaX)),
              y: Math.max(0, Math.min(initialBox.y + initialBox.height - 50, initialBox.y + deltaY)),
              width: Math.max(50, Math.min(screenshotData.width - initialBox.x, initialBox.width - deltaX)),
              height: Math.max(50, Math.min(screenshotData.height - initialBox.y, initialBox.height - deltaY))
            };
            break;
          case 'top-right':
            newBox = {
              x: initialBox.x,
              y: Math.max(0, Math.min(initialBox.y + initialBox.height - 50, initialBox.y + deltaY)),
              width: Math.max(50, Math.min(screenshotData.width - initialBox.x, initialBox.width + deltaX)),
              height: Math.max(50, Math.min(screenshotData.height - initialBox.y, initialBox.height - deltaY))
            };
            break;
          case 'bottom-left':
            newBox = {
              x: Math.max(0, Math.min(initialBox.x + initialBox.width - 50, initialBox.x + deltaX)),
              y: initialBox.y,
              width: Math.max(50, Math.min(screenshotData.width - initialBox.x, initialBox.width - deltaX)),
              height: Math.max(50, Math.min(screenshotData.height - initialBox.y, initialBox.height + deltaY))
            };
            break;
          case 'bottom-right':
            newBox = {
              x: initialBox.x,
              y: initialBox.y,
              width: Math.max(50, Math.min(screenshotData.width - initialBox.x, initialBox.width + deltaX)),
              height: Math.max(50, Math.min(screenshotData.height - initialBox.y, initialBox.height + deltaY))
            };
            break;
          case 'top':
            newBox = {
              x: initialBox.x,
              y: Math.max(0, Math.min(initialBox.y + initialBox.height - 50, initialBox.y + deltaY)),
              width: initialBox.width,
              height: Math.max(50, Math.min(screenshotData.height - initialBox.y, initialBox.height - deltaY))
            };
            break;
          case 'bottom':
            newBox = {
              x: initialBox.x,
              y: initialBox.y,
              width: initialBox.width,
              height: Math.max(50, Math.min(screenshotData.height - initialBox.y, initialBox.height + deltaY))
            };
            break;
          case 'left':
            newBox = {
              x: Math.max(0, Math.min(initialBox.x + initialBox.width - 50, initialBox.x + deltaX)),
              y: initialBox.y,
              width: Math.max(50, Math.min(screenshotData.width - initialBox.x, initialBox.width - deltaX)),
              height: initialBox.height
            };
            break;
          case 'right':
            newBox = {
              x: initialBox.x,
              y: initialBox.y,
              width: Math.max(50, Math.min(screenshotData.width - initialBox.x, initialBox.width + deltaX)),
              height: initialBox.height
            };
            break;
        }
      }
      
      // 更新引用中的当前状态
      Object.assign(currentBoxRef, newBox);
      
      // 使用函数式更新确保我们总是基于最新状态进行更新
      setSelectionBox(newBox);
    };
    
    const handleMouseUp = () => {
      // 清理事件监听器
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('mouseleave', handleMouseUp);
    };
    
    // 添加事件监听器到document而不是组件
    document.addEventListener('mousemove', handleMouseMove, { passive: false });
    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('mouseleave', handleMouseUp);
  }

  // 拍照功能
  async function handleShoot() {
    if (!videoRef.current) return;
    setRespText('');
    setBusy(true);
    try {
      const blob = await captureCompressedJpeg(videoRef.current);
      setLastSizeKB(Math.round(blob.size / 1024));

      const dataUrl = await new Promise((res) => {
        const fr = new FileReader();
        fr.onload = () => res(fr.result);
        fr.readAsDataURL(blob);
      });
      const base64 = String(dataUrl).split(',')[1];

      await sendImageToAI(base64);
    } catch (e) {
      setRespText('Error: ' + (e?.message || String(e)));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ maxWidth: 920, margin: '0 auto', padding: '24px' }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h1>AI Solver (Hunyuan)</h1>
        <a 
          href="/docs/ai-solver-tutorial" 
          style={{
            padding: "8px 16px",
            backgroundColor: "#000000",
            color: "#ffffff",
            textDecoration: "none",
            borderRadius: "10px",
            fontSize: "14px",
            fontWeight: "500",
            border: "1px solid #000000",
            transition: "all 0.2s ease"
          }}
          onMouseOver={(e) => {
            e.target.style.backgroundColor = "#333333";
            e.target.style.borderColor = "#333333";
            e.target.style.transform = "translateY(-1px)";
          }}
          onMouseOut={(e) => {
            e.target.style.backgroundColor = "#000000";
            e.target.style.borderColor = "#000000";
            e.target.style.transform = "translateY(0)";
          }}
        >
           Tutorial
        </a>
      </div>
      <p>Supports camera capture, screen capture, and text questions. For security, the page does not accept any keys.</p>
      <p style={{ fontSize: 14, color: '#666' }}>Tip: In text mode, type <code>/preset name</code> to quickly switch presets, e.g. <code>/preset Math Problem Solver</code></p>

      <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div>
          {/* Input Mode Selection */}
          <fieldset style={{ border: '1px solid #ddd', borderRadius: 8, padding: 12, marginBottom: 16 }}>
            <legend>Input Mode</legend>
            <label style={{ display: 'block', marginBottom: 8 }}>
              <input 
                type="radio" 
                value="camera" 
                checked={captureMode === 'camera'} 
                onChange={e => setCaptureMode(e.target.value)} 
              />
              <span style={{ marginLeft: 8 }}> Camera Capture</span>
            </label>
            <label style={{ display: 'block', marginBottom: 8 }}>
              <input 
                type="radio" 
                value="screenshot" 
                checked={captureMode === 'screenshot'} 
                onChange={e => setCaptureMode(e.target.value)} 
              />
              <span style={{ marginLeft: 8 }}> Screen Capture</span>
            </label>
            <label style={{ display: 'block' }}>
              <input 
                type="radio" 
                value="text" 
                checked={captureMode === 'text'} 
                onChange={e => setCaptureMode(e.target.value)} 
              />
              <span style={{ marginLeft: 8 }}> Text Question</span>
            </label>
          </fieldset>

          {/* Camera Preview */}
          {captureMode === 'camera' && (
            <div>
              <video ref={videoRef} autoPlay playsInline style={{ width: 320, background: '#000', borderRadius: 8 }} />
              <div style={{ marginTop: 8, color: '#666' }}>
                {ready ? 'Camera ready' : error ? `Camera error: ${error}` : 'Requesting camera permission…'}
              </div>
              <button onClick={handleShoot} disabled={!ready || busy} style={{ marginTop: 12, padding: '8px 16px', fontSize: 14 }}>
                {busy ? 'Processing…' : ' Capture and Solve'}
              </button>
            </div>
          )}

          {/* Screenshot Mode */}
          {captureMode === 'screenshot' && (
            <div>
              {!isSelecting ? (
                // 初始状态 - 显示截图按钮
                <>
                  <div style={{ width: 320, height: 240, background: '#f5f5f5', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', border: '2px dashed #ccc' }}>
                    <div style={{ textAlign: 'center', color: '#666' }}>
                      <div style={{ fontSize: 48, marginBottom: 8 }}> </div>
                      <div>Click the button below to start screenshot</div>
                    </div>
                  </div>
                  <div style={{ marginTop: 8, color: '#666' }}>
                    Screenshot mode: You can select a specific area after capturing
                  </div>
                  <button onClick={handleScreenshot} disabled={busy} style={{ marginTop: 12, padding: '8px 16px', fontSize: 14 }}>
                    {busy ? 'Processing…' : ' Capture Screen'}
                  </button>
                </>
              ) : (
                // 选择状态 - 显示截图和选择框
                <>
                  <div style={{ position: 'relative', width: 320, height: 240, border: '2px solid #000000', borderRadius: 8, overflow: 'hidden' }}>
                    <img 
                      src={screenshotData?.dataUrl} 
                      alt="Screenshot" 
                      style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                      onMouseDown={(e) => handleSelectionDrag(e, 'move')}
                    />
                    {selectionBox && screenshotData && (
                      <div
                        style={{
                          position: 'absolute',
                          left: `${(selectionBox.x / screenshotData.width) * 100}%`,
                          top: `${(selectionBox.y / screenshotData.height) * 100}%`,
                          width: `${(selectionBox.width / screenshotData.width) * 100}%`,
                          height: `${(selectionBox.height / screenshotData.height) * 100}%`,
                          border: '2px solid #333333',
                          backgroundColor: 'rgba(0, 0, 0, 0.06)',
                          cursor: 'move',
                          boxSizing: 'border-box'
                        }}
                        onMouseDown={(e) => {
                          e.stopPropagation();
                          handleSelectionDrag(e, 'move');
                        }}
                      >
                        {/* 八个方向的调整大小手柄 */}
                        {/* 左上角 */}
                        <div
                          style={{
                            position: 'absolute',
                            left: -4,
                            top: -4,
                            width: 8,
                            height: 8,
                            backgroundColor: '#333333',
                            cursor: 'nw-resize',
                            borderRadius: '50%'
                          }}
                          onMouseDown={(e) => {
                            e.stopPropagation();
                            handleSelectionDrag(e, 'resize', 'top-left');
                          }}
                        />
                        {/* 上边 */}
                        <div
                          style={{
                            position: 'absolute',
                            left: '50%',
                            top: -4,
                            width: 8,
                            height: 8,
                            backgroundColor: '#333333',
                            cursor: 'n-resize',
                            borderRadius: '50%',
                            transform: 'translateX(-50%)'
                          }}
                          onMouseDown={(e) => {
                            e.stopPropagation();
                            handleSelectionDrag(e, 'resize', 'top');
                          }}
                        />
                        {/* 右上角 */}
                        <div
                          style={{
                            position: 'absolute',
                            right: -4,
                            top: -4,
                            width: 8,
                            height: 8,
                            backgroundColor: '#333333',
                            cursor: 'ne-resize',
                            borderRadius: '50%'
                          }}
                          onMouseDown={(e) => {
                            e.stopPropagation();
                            handleSelectionDrag(e, 'resize', 'top-right');
                          }}
                        />
                        {/* 右边 */}
                        <div
                          style={{
                            position: 'absolute',
                            right: -4,
                            top: '50%',
                            width: 8,
                            height: 8,
                            backgroundColor: '#333333',
                            cursor: 'e-resize',
                            borderRadius: '50%',
                            transform: 'translateY(-50%)'
                          }}
                          onMouseDown={(e) => {
                            e.stopPropagation();
                            handleSelectionDrag(e, 'resize', 'right');
                          }}
                        />
                        {/* 右下角 */}
                        <div
                          style={{
                            position: 'absolute',
                            right: -4,
                            bottom: -4,
                            width: 8,
                            height: 8,
                            backgroundColor: '#333333',
                            cursor: 'se-resize',
                            borderRadius: '50%'
                          }}
                          onMouseDown={(e) => {
                            e.stopPropagation();
                            handleSelectionDrag(e, 'resize', 'bottom-right');
                          }}
                        />
                        {/* 下边 */}
                        <div
                          style={{
                            position: 'absolute',
                            left: '50%',
                            bottom: -4,
                            width: 8,
                            height: 8,
                            backgroundColor: '#333333',
                            cursor: 's-resize',
                            borderRadius: '50%',
                            transform: 'translateX(-50%)'
                          }}
                          onMouseDown={(e) => {
                            e.stopPropagation();
                            handleSelectionDrag(e, 'resize', 'bottom');
                          }}
                        />
                        {/* 左下角 */}
                        <div
                          style={{
                            position: 'absolute',
                            left: -4,
                            bottom: -4,
                            width: 8,
                            height: 8,
                            backgroundColor: '#333333',
                            cursor: 'sw-resize',
                            borderRadius: '50%'
                          }}
                          onMouseDown={(e) => {
                            e.stopPropagation();
                            handleSelectionDrag(e, 'resize', 'bottom-left');
                          }}
                        />
                        {/* 左边 */}
                        <div
                          style={{
                            position: 'absolute',
                            left: -4,
                            top: '50%',
                            width: 8,
                            height: 8,
                            backgroundColor: '#333333',
                            cursor: 'w-resize',
                            borderRadius: '50%',
                            transform: 'translateY(-50%)'
                          }}
                          onMouseDown={(e) => {
                            e.stopPropagation();
                            handleSelectionDrag(e, 'resize', 'left');
                          }}
                        />
                      </div>
                    )}
                  </div>
                  <div style={{ marginTop: 8, color: '#666', fontSize: 12 }}>
                     Selection box shows selected area. Drag to move, drag corner to resize.
                  </div>
                  <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
                    <button 
                      onClick={handleConfirmSelection} 
                      disabled={busy} 
                      style={{ padding: '8px 16px', fontSize: 14, backgroundColor: '#6c6c70', color: 'white', border: '1px solid #6c6c70', borderRadius: 4, cursor: 'pointer' }}
>
                      {busy ? 'Processing…' : 'Analyze Selected Area'}
                    </button>
                    <button 
                      onClick={handleCancelSelection} 
                      disabled={busy}
                      style={{ padding: '8px 16px', fontSize: 14, backgroundColor: '#6c6c70', color: 'white', border: '1px solid #6c6c70', borderRadius: 4, cursor: 'not-allowed' }}
                    >
                       Cancel
                    </button>
                  </div>
                </>
              )}
            </div>
          )}

          {/* Text Question Mode */}
          {captureMode === 'text' && (
            <div>
              <div style={{ width: 320, minHeight: 240, background: '#f9f9f9', borderRadius: 8, padding: 16, border: '1px solid #ddd' }}>
                <div style={{ marginBottom: 12, color: '#666', fontSize: 14 }}> Text Question Mode</div>
                <textarea 
                  value={textInput}
                  onChange={e => setTextInput(e.target.value)}
                  placeholder="Please enter your question, for example:&#10;• Explain the basic principles of quantum mechanics&#10;• Write a Python sorting algorithm&#10;• Type /preset to see available presets"
                  style={{ 
                    width: '100%', 
                    height: 160, 
                    border: '1px solid #ccc', 
                    borderRadius: 4, 
                    padding: 8, 
                    fontSize: 14,
                    resize: 'vertical',
                    fontFamily: 'inherit'
                  }}
                />
              </div>
              <div style={{ marginTop: 8, color: '#666' }}>
                Enter your question directly and the AI will provide a detailed answer
              </div>
              <button 
                onClick={handleTextQuestion} 
                disabled={busy || !textInput.trim()} 
                style={{ marginTop: 12, padding: '8px 16px', fontSize: 14 }}
              >
                {busy ? 'Thinking…' : ' Ask and Get Answer'}
              </button>
            </div>
          )}

          {lastSizeKB != null && (
            <div style={{ marginTop: 6, fontSize: 12, color: '#999' }}>Last image size: {lastSizeKB} KB</div>
          )}
        </div>

        <div style={{ flex: 1, minWidth: 280 }}>
          <fieldset style={{ border: '1px solid #e5e5ea', borderRadius: 8, padding: 12, marginBottom: 16, background: '#f5f5f7' }}>
            <legend>API Settings</legend>
            
            {/* Use Default API Button */}
            <div style={{ marginBottom: 12 }}>
              <button 
                type="button"
                onClick={() => setUseDefaultApi(!useDefaultApi)}
                style={{
                  padding: '6px 12px',
                  fontSize: 12,
                  backgroundColor: useDefaultApi ? '#4CAF50' : '#f0f0f0',
                  color: useDefaultApi ? 'white' : '#333',
                  border: '1px solid #ccc',
                  borderRadius: 4,
                  cursor: 'pointer'
                }}
              >
                {useDefaultApi ? '✓ Using Default API' : 'Use Default API'}
              </button>
              <div style={{ fontSize: 11, color: '#666', marginTop: 4 }}>
                {useDefaultApi
                  ? `Default API enabled → ${defaultApiEndpoint}`
                  : `Click to send requests to ${defaultApiEndpoint}`}
              </div>
            </div>
            
            <label>Custom API URL (optional)<br />
              <input 
                value={apiUrl} 
                onChange={e => setApiUrl(e.target.value)} 
                placeholder="https://your-api.example.com/api/solve" 
                style={{ width: '100%' }} 
                disabled={useDefaultApi}
              />
            </label>
            <div style={{ color: '#345', fontSize: 12, marginTop: 8 }}>
              {useDefaultApi 
                ? 'Using default API. Disable to enter custom API.' 
                : 'Enter your API endpoint or enable default API above.'}
            </div>
          </fieldset>

          <fieldset style={{ border: '1px solid #ddd', borderRadius: 8, padding: 12 }}>
            <legend>Request Parameters</legend>
            <label>Model<br />
              <input value={model} onChange={e => setModel(e.target.value)} style={{ width: '100%' }} />
            </label>
            
            {/* Prompt预设选择 */}
            <div style={{ marginTop: 8 }}>
              <label>PromptPreset<br />
                <select 
                  value={selectedPreset} 
                  onChange={(e) => {
                    setSelectedPreset(e.target.value);
                    const preset = promptPresets.find(p => p.id === e.target.value);
                    if (preset) {
                      if (preset.id === 'custom') {
                        setQuestion('');
                      } else {
                        setQuestion(preset.prompt);
                      }
                    }
                  }}
                  style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid #ccc' }}
                >
                  {promptPresets.map(preset => (
                    <option key={preset.id} value={preset.id}>{preset.name}</option>
                  ))}
                </select>
              </label>
              <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
                {promptPresets.find(p => p.id === selectedPreset)?.description || ''}
              </div>
            </div>
            
            {selectedPreset === 'custom' ? (
              <label style={{ display: 'block', marginTop: 8 }}>Question (for image mode)<br />
                <textarea 
                  value={question} 
                  onChange={e => {
                    setQuestion(e.target.value);
                  }} 
                  rows={3} 
                  style={{ width: '100%' }} 
                />
              </label>
            ) : (
              <div style={{ marginTop: 12, fontSize: 12, color: '#666' }}>
                Preset instructions are applied automatically based on the selected mode.
              </div>
            )}
          </fieldset>
        </div>
      </div>

      <div style={{ marginTop: 16 }}>
        <h3>Response</h3>
        <div style={{ 
          whiteSpace: 'pre-wrap', 
          background: '#f8f9fa', 
          color: '#212529', 
          padding: 16, 
          borderRadius: 8, 
          maxHeight: 400, 
          overflow: 'auto',
          border: '1px solid #dee2e6',
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
          fontSize: '16px',
          lineHeight: 1.5
        }}>
          {respText}
        </div>
        <div style={{ marginTop: 8, textAlign: 'right' }}>
          <button 
            onClick={() => {
              if (window.lastFullResponse) {
                setRespText(JSON.stringify(window.lastFullResponse, null, 2));
              }
            }}
            style={{ 
              padding: '4px 8px', 
              fontSize: 12, 
              background: 'transparent', 
              border: '1px solid #ccc', 
              borderRadius: 4, 
              cursor: 'pointer',
              color: '#666'
            }}
          >
            Show Raw JSON
          </button>
        </div>
      </div>
    </div>
  );
}
