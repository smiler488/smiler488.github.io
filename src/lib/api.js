const DEFAULT_SYSTEM_PROMPT =
  "Be accurate, concise, and explicit about uncertainty. Follow the requested output format.";

export const AI_PROVIDER_PRESETS = [
  {
    id: "demo",
    name: "Local demo",
    protocol: "mock",
    endpoint: "",
    description:
      "Runs a local sample response without sending data or requiring a key.",
    models: [{ id: "local-demo", label: "Local demo" }],
    lastVerifiedAt: "2026-07-15",
  },
  {
    id: "openai",
    name: "OpenAI",
    protocol: "openai",
    endpoint: "https://api.openai.com/v1/chat/completions",
    endpointLocked: true,
    supportsJsonMode: true,
    description:
      "Official OpenAI Chat Completions endpoint. Direct browser use may be blocked.",
    models: [
      { id: "gpt-5.6-terra", label: "GPT-5.6 Terra", vision: true },
      { id: "gpt-5.6-luna", label: "GPT-5.6 Luna", vision: true },
      { id: "gpt-5.6", label: "GPT-5.6", vision: true },
    ],
    lastVerifiedAt: "2026-07-15",
  },
  {
    id: "anthropic",
    name: "Anthropic Claude",
    protocol: "anthropic",
    endpoint: "https://api.anthropic.com/v1/messages",
    endpointLocked: true,
    description:
      "Official Anthropic Messages endpoint. Some organizations disable browser CORS.",
    models: [
      { id: "claude-sonnet-5", label: "Claude Sonnet 5", vision: true },
      {
        id: "claude-haiku-4-5-20251001",
        label: "Claude Haiku 4.5",
        vision: true,
      },
      { id: "claude-opus-4-8", label: "Claude Opus 4.8", vision: true },
    ],
    lastVerifiedAt: "2026-07-15",
  },
  {
    id: "gemini",
    name: "Google Gemini",
    protocol: "gemini",
    endpoint:
      "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    endpointLocked: true,
    supportsJsonMode: true,
    description:
      "Official Google generateContent endpoint. Use a restricted key for testing only.",
    models: [
      { id: "gemini-3.5-flash", label: "Gemini 3.5 Flash", vision: true },
      {
        id: "gemini-3.1-flash-lite",
        label: "Gemini 3.1 Flash Lite",
        vision: true,
      },
      {
        id: "gemini-3.1-pro-preview",
        label: "Gemini 3.1 Pro Preview",
        vision: true,
      },
    ],
    lastVerifiedAt: "2026-07-15",
  },
  {
    id: "deepseek",
    name: "DeepSeek",
    protocol: "openai",
    endpoint: "https://api.deepseek.com/chat/completions",
    endpointLocked: true,
    description: "DeepSeek OpenAI-compatible API.",
    models: [
      { id: "deepseek-v4-flash", label: "DeepSeek V4 Flash" },
      { id: "deepseek-v4-pro", label: "DeepSeek V4 Pro" },
    ],
    lastVerifiedAt: "2026-07-15",
  },
  {
    id: "qwen",
    name: "Alibaba Qwen (China)",
    protocol: "openai",
    endpoint:
      "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    endpointLocked: true,
    description:
      "Alibaba Cloud Model Studio Beijing endpoint. Keys are region-bound.",
    models: [
      { id: "qwen3.7-plus", label: "Qwen 3.7 Plus" },
      { id: "qwen3.7-max", label: "Qwen 3.7 Max" },
      { id: "qwen3.6-flash", label: "Qwen 3.6 Flash" },
    ],
    lastVerifiedAt: "2026-07-15",
  },
  {
    id: "hunyuan",
    name: "Tencent Hunyuan",
    protocol: "openai",
    endpoint: "https://tokenhub.tencentmaas.com/v1/chat/completions",
    endpointLocked: true,
    description:
      "Tencent TokenHub Guangzhou endpoint. Keys can be limited by model and quota.",
    models: [
      { id: "hy3", label: "Hunyuan HY 3" },
      { id: "hy3-preview", label: "Hunyuan HY 3 Preview" },
    ],
    lastVerifiedAt: "2026-07-15",
  },
  {
    id: "custom",
    name: "Custom compatible API",
    protocol: "openai",
    endpoint: "",
    description:
      "Advanced: an HTTPS OpenAI-compatible endpoint that you trust.",
    models: [],
    lastVerifiedAt: null,
  },
];

export function getAIProvider(providerId) {
  return (
    AI_PROVIDER_PRESETS.find((provider) => provider.id === providerId) ||
    AI_PROVIDER_PRESETS[0]
  );
}

export function createDefaultAIConfig() {
  return {
    provider: "demo",
    endpoint: "",
    model: "local-demo",
    apiKey: "",
  };
}

export function changeAIProvider(config, providerId) {
  const provider = getAIProvider(providerId);
  return {
    ...config,
    provider: provider.id,
    endpoint: provider.endpoint,
    model: provider.models[0]?.id || "",
    apiKey: "",
  };
}

function parseDataUrl(value) {
  if (!value || typeof value !== "string") return null;
  const match = value.match(/^data:([^;,]+);base64,(.+)$/);
  if (!match) return null;
  return { mediaType: match[1], data: match[2] };
}

function normalizeImage(input) {
  if (input.imageUrl?.trim()) {
    const value = input.imageUrl.trim();
    const parsed = parseDataUrl(value);
    return parsed ? { ...parsed, url: value } : { url: value };
  }
  if (input.imageBase64) {
    const value = input.imageBase64.startsWith("data:")
      ? input.imageBase64
      : `data:image/jpeg;base64,${input.imageBase64}`;
    return { ...parseDataUrl(value), url: value };
  }
  return null;
}

function buildOpenAIRequest(config, provider, input, options) {
  const image = normalizeImage(input);
  const content = [];
  const question =
    (input.question || "").trim() || "Please analyze the provided input.";
  content.push({ type: "text", text: question });
  if (image?.url) {
    content.push({ type: "image_url", image_url: { url: image.url } });
  }

  const body = {
    model: config.model,
    messages: [
      {
        role: "system",
        content: options.systemPrompt || DEFAULT_SYSTEM_PROMPT,
      },
      { role: "user", content: image?.url ? content : question },
    ],
    stream: false,
  };

  if (options.jsonMode && provider.supportsJsonMode) {
    body.response_format = { type: "json_object" };
  }

  return {
    url: config.endpoint,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.apiKey}`,
    },
    body,
  };
}

function buildAnthropicRequest(config, input, options) {
  const image = normalizeImage(input);
  const content = [];
  if (image?.data) {
    content.push({
      type: "image",
      source: {
        type: "base64",
        media_type: image.mediaType || "image/jpeg",
        data: image.data,
      },
    });
  } else if (image?.url) {
    content.push({ type: "image", source: { type: "url", url: image.url } });
  }
  content.push({
    type: "text",
    text: (input.question || "").trim() || "Please analyze the provided input.",
  });

  return {
    url: config.endpoint,
    headers: {
      "Content-Type": "application/json",
      "x-api-key": config.apiKey,
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
    },
    body: {
      model: config.model,
      max_tokens: options.maxTokens || 4096,
      system: options.systemPrompt || DEFAULT_SYSTEM_PROMPT,
      messages: [{ role: "user", content }],
    },
  };
}

function buildGeminiRequest(config, input, options) {
  const image = normalizeImage(input);
  const parts = [];
  if (image?.data) {
    parts.push({
      inline_data: {
        mime_type: image.mediaType || "image/jpeg",
        data: image.data,
      },
    });
  }
  parts.push({
    text: (input.question || "").trim() || "Please analyze the provided input.",
  });

  const body = {
    system_instruction: {
      parts: [{ text: options.systemPrompt || DEFAULT_SYSTEM_PROMPT }],
    },
    contents: [{ role: "user", parts }],
  };
  if (options.jsonMode) {
    body.generationConfig = { responseMimeType: "application/json" };
  }

  return {
    url: config.endpoint.replace("{model}", encodeURIComponent(config.model)),
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": config.apiKey,
    },
    body,
  };
}

export function buildAIRequest(config, input, options = {}) {
  const provider = getAIProvider(config.provider);
  if (provider.protocol === "anthropic") {
    return buildAnthropicRequest(config, input, options);
  }
  if (provider.protocol === "gemini") {
    return buildGeminiRequest(config, input, options);
  }
  return buildOpenAIRequest(config, provider, input, options);
}

export function extractAssistantText(data) {
  if (Array.isArray(data?.choices) && data.choices.length > 0) {
    const content = data.choices[0]?.message?.content;
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
      return content
        .map((part) => part?.text || "")
        .filter(Boolean)
        .join("\n");
    }
  }
  if (Array.isArray(data?.content)) {
    return data.content
      .map((part) => part?.text || "")
      .filter(Boolean)
      .join("\n");
  }
  if (Array.isArray(data?.candidates)) {
    return (data.candidates[0]?.content?.parts || [])
      .map((part) => part?.text || "")
      .filter(Boolean)
      .join("\n");
  }
  if (data?.Response && Array.isArray(data.Response.Choices)) {
    return data.Response.Choices[0]?.Message?.Content || "";
  }
  return typeof data === "string" ? data : JSON.stringify(data, null, 2);
}

function extractUsage(data) {
  if (data?.usage) return data.usage;
  if (data?.usageMetadata) return data.usageMetadata;
  if (data?.Response?.Usage) return data.Response.Usage;
  return null;
}

export async function requestAI(config, input, options = {}) {
  const provider = getAIProvider(config.provider);
  if (provider.protocol === "mock") {
    const response = await postJson(
      `mock://${options.mockTag || "ai-solver"}`,
      {
        ...input,
        model: config.model,
      }
    );
    const raw = await response.json();
    return { text: extractAssistantText(raw), raw, usage: extractUsage(raw) };
  }

  const effectiveEndpoint = provider.endpointLocked
    ? provider.endpoint
    : config.endpoint;
  if (!effectiveEndpoint?.trim()) {
    throw new Error("Enter an API endpoint before sending a request.");
  }
  if (!config.model?.trim()) {
    throw new Error("Choose or enter a model before sending a request.");
  }
  if (!config.apiKey?.trim()) {
    throw new Error("Enter the API key for the selected provider.");
  }

  const selectedModel = provider.models.find(
    (model) => model.id === config.model.trim()
  );
  if (options.requireVision && selectedModel && !selectedModel.vision) {
    throw new Error(
      `${selectedModel.label} is not marked as an image-capable model.`
    );
  }

  const request = buildAIRequest(
    {
      ...config,
      endpoint: effectiveEndpoint.trim(),
      model: config.model.trim(),
      apiKey: config.apiKey.trim(),
    },
    input,
    options
  );
  assertSafeEndpoint(request.url);
  const response = await fetch(request.url, {
    method: "POST",
    headers: request.headers,
    body: JSON.stringify(request.body),
    signal: options.signal,
  });

  if (!response.ok) {
    const rawError = await response.text().catch(() => "");
    const detail = sanitizeErrorDetail(rawError, config.apiKey);
    throw new Error(
      `${provider.name} returned ${response.status}${
        detail ? `: ${detail}` : ""
      }`
    );
  }

  const raw = await response.json();
  return { text: extractAssistantText(raw), raw, usage: extractUsage(raw) };
}

function assertSafeEndpoint(value) {
  let url;
  try {
    url = new URL(value);
  } catch {
    throw new Error("Enter a valid absolute API endpoint.");
  }
  const localDevelopmentHost = ["localhost", "127.0.0.1", "[::1]"].includes(
    url.hostname
  );
  if (
    url.protocol !== "https:" &&
    !(url.protocol === "http:" && localDevelopmentHost)
  ) {
    throw new Error(
      "API endpoints must use HTTPS (HTTP is allowed only on localhost)."
    );
  }
  if (url.username || url.password) {
    throw new Error("Do not place credentials in the API endpoint URL.");
  }
}

function sanitizeErrorDetail(rawError, apiKey) {
  let detail = String(rawError || "")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (apiKey) detail = detail.split(apiKey).join("[redacted]");
  return detail
    .replace(/(?:sk|key|token)-[A-Za-z0-9_-]{8,}/gi, "[redacted]")
    .replace(/AIza[A-Za-z0-9_-]{20,}/g, "[redacted]")
    .slice(0, 280);
}

export async function postJson(url, json, extraHeaders = {}) {
  if (typeof url === "string" && url.startsWith("mock://")) {
    const tag = url.slice("mock://".length);
    if (tag === "ai-data-visualizer") {
      const mockBody = {
        summary:
          "Sample analysis: values trend upward overall, with the final group showing the strongest result.",
        insights: [
          "The dataset contains a clear overall trend.",
          "Group-level differences are large enough to visualize.",
          "Review sample size and uncertainty before drawing a causal conclusion.",
        ],
        chart_option: {
          title: { text: "Demo data overview", left: "center" },
          tooltip: { position: "top" },
          grid: { left: "5%", right: "5%", top: "12%", bottom: "18%" },
          xAxis: { type: "category", data: ["North", "South", "East", "West"] },
          yAxis: { type: "value" },
          series: [{ name: "Value", type: "bar", data: [120, 135, 150, 168] }],
        },
      };
      const payload = {
        choices: [{ message: { content: JSON.stringify(mockBody, null, 2) } }],
      };
      return mockResponse(payload);
    }
    if (tag === "journal-selector" || tag === "journal-scout") {
      const mockBody = {
        overview: {
          abstract_summary: json?.question?.slice(0, 120) || "N/A",
          alignment_summary:
            "Local demo response. Add your own provider key for live journal analysis.",
        },
        journals: [
          {
            journal_name: "Demo Journal of Digital Agriculture",
            publisher: "Demo Publisher",
            discipline_scope: "Plant phenotyping and agricultural AI",
            impact_factor_2024: "N/A",
            jcr_quartile: "N/A",
            acceptance_rate: "N/A",
            initial_review_weeks: 6,
            oa_type: "Unknown",
            apc_usd: null,
            submission_advice:
              "Validate all journal metrics on the official journal website.",
            warning_status: "Demo data — not publication advice",
          },
        ],
        notes:
          "This local demo intentionally avoids presenting fabricated current metrics.",
      };
      const payload = {
        choices: [{ message: { content: JSON.stringify(mockBody, null, 2) } }],
      };
      return mockResponse(payload);
    }
    if (tag === "ai-solver" || tag === "solver") {
      const q = json?.question || "No question provided";
      const hasImage = !!json?.imageBase64 || !!json?.imageUrl;
      const content = `${
        hasImage ? "Demo vision analysis" : "Demo text analysis"
      }\n\nUser question:\n${q}\n\nThis is an offline sample. Select a provider and enter your own key for a live response.`;
      const payload = {
        choices: [{ message: { content } }],
        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
        mock: true,
      };
      return mockResponse(payload);
    }
  }

  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...extraHeaders },
    body: JSON.stringify(json),
  });
}

function mockResponse(payload) {
  return {
    ok: true,
    status: 200,
    json: async () => payload,
    text: async () => JSON.stringify(payload),
    headers: new Map([["content-type", "application/json"]]),
  };
}
