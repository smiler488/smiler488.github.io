import {
  AI_CATALOG_CHECKED_AT,
  AI_PROVIDER_GROUPS,
  AI_PROVIDER_PRESETS,
} from "./aiProviders.js";
import { IS_ZH, makeToolText } from "./i18n/toolText.js";
import ZH from "./api_zh.js";

const tx = makeToolText(ZH);

const DEFAULT_SYSTEM_PROMPT =
  "Be accurate, concise, and explicit about uncertainty. Follow the requested output format.";

export { AI_CATALOG_CHECKED_AT, AI_PROVIDER_GROUPS, AI_PROVIDER_PRESETS };

/** Provider names and descriptions are stored as { en, zh }. */
export function localized(value) {
  if (value && typeof value === "object") return IS_ZH ? value.zh : value.en;
  return value || "";
}

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

function questionText(input) {
  return (input.question || "").trim() || "Please analyze the provided input.";
}

function bearerHeaders(apiKey) {
  const headers = { "Content-Type": "application/json" };
  if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
  return headers;
}

function buildOpenAIRequest(config, provider, input, options) {
  const image = normalizeImage(input);
  const question = questionText(input);
  const content = [{ type: "text", text: question }];
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
  if (options.maxTokens) body.max_tokens = options.maxTokens;
  if (options.jsonMode && provider.supportsJsonMode) {
    body.response_format = { type: "json_object" };
  }

  return { url: config.endpoint, headers: bearerHeaders(config.apiKey), body };
}

// OpenAI Responses API (OpenAI, xAI, Meta).
function buildResponsesRequest(config, provider, input, options) {
  const image = normalizeImage(input);
  const content = [{ type: "input_text", text: questionText(input) }];
  if (image?.url) content.push({ type: "input_image", image_url: image.url });

  const body = {
    model: config.model,
    instructions: options.systemPrompt || DEFAULT_SYSTEM_PROMPT,
    input: [{ role: "user", content }],
    store: false,
  };
  if (options.maxTokens) body.max_output_tokens = options.maxTokens;
  if (options.jsonMode && provider.supportsJsonMode) {
    body.text = { format: { type: "json_object" } };
  }

  return { url: config.endpoint, headers: bearerHeaders(config.apiKey), body };
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
  content.push({ type: "text", text: questionText(input) });

  const headers = {
    "Content-Type": "application/json",
    "x-api-key": config.apiKey,
    "anthropic-version": "2023-06-01",
  };
  // Anthropic requires this opt-in for browser calls; other servers that
  // speak the Messages format may reject the unknown header in preflight.
  if (/^https:\/\/api\.anthropic\.com\//.test(config.endpoint)) {
    headers["anthropic-dangerous-direct-browser-access"] = "true";
  }

  return {
    url: config.endpoint,
    headers,
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
  parts.push({ text: questionText(input) });

  const body = {
    system_instruction: {
      parts: [{ text: options.systemPrompt || DEFAULT_SYSTEM_PROMPT }],
    },
    contents: [{ role: "user", parts }],
  };
  const generationConfig = {};
  if (options.jsonMode) generationConfig.responseMimeType = "application/json";
  if (options.maxTokens) generationConfig.maxOutputTokens = options.maxTokens;
  if (Object.keys(generationConfig).length) {
    body.generationConfig = generationConfig;
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
  switch (provider.protocol) {
    case "anthropic":
      return buildAnthropicRequest(config, input, options);
    case "gemini":
      return buildGeminiRequest(config, input, options);
    case "openai-responses":
      return buildResponsesRequest(config, provider, input, options);
    default:
      return buildOpenAIRequest(config, provider, input, options);
  }
}

function joinText(parts) {
  return parts
    .map((part) => part?.text || "")
    .filter(Boolean)
    .join("\n");
}

export function extractAssistantText(data) {
  if (Array.isArray(data?.choices) && data.choices.length > 0) {
    const content = data.choices[0]?.message?.content;
    if (typeof content === "string") return content;
    if (Array.isArray(content)) return joinText(content);
  }
  // Responses API: message items carry output_text parts.
  if (Array.isArray(data?.output)) {
    return joinText(
      data.output
        .filter((item) => item?.type === "message")
        .flatMap((item) => item.content || [])
        .filter((part) => part?.type === "output_text")
    );
  }
  if (Array.isArray(data?.content)) {
    return joinText(data.content.filter((part) => part?.type !== "thinking"));
  }
  if (Array.isArray(data?.candidates)) {
    return joinText(
      (data.candidates[0]?.content?.parts || []).filter(
        (part) => !part?.thought
      )
    );
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

  const providerName = localized(provider.name);
  const effectiveEndpoint = provider.endpointLocked
    ? provider.endpoint
    : config.endpoint;
  if (!effectiveEndpoint?.trim()) {
    throw new Error(tx("Enter an API endpoint before sending a request."));
  }
  if (!config.model?.trim()) {
    throw new Error(tx("Choose or enter a model before sending a request."));
  }
  if (!config.apiKey?.trim() && !provider.keyOptional) {
    throw new Error(tx("Enter the API key for the selected provider."));
  }

  const selectedModel = provider.models.find(
    (model) => model.id === config.model.trim()
  );
  if (options.requireVision && selectedModel && !selectedModel.vision) {
    throw new Error(
      tx("{0} does not accept image input.", selectedModel.label)
    );
  }

  assertSafeEndpoint(effectiveEndpoint.trim());
  const request = buildAIRequest(
    {
      ...config,
      endpoint: effectiveEndpoint.trim(),
      model: config.model.trim(),
      apiKey: (config.apiKey || "").trim(),
    },
    input,
    options
  );
  assertSafeEndpoint(request.url);

  let response;
  try {
    response = await fetch(request.url, {
      method: "POST",
      headers: request.headers,
      body: JSON.stringify(request.body),
      signal: options.signal,
    });
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    throw new Error(
      tx(
        "Could not reach {0}. The browser may have blocked the request (CORS), or the server is offline.",
        providerName
      )
    );
  }

  if (!response.ok) {
    const rawError = await response.text().catch(() => "");
    const detail = sanitizeErrorDetail(rawError, config.apiKey);
    throw new Error(
      tx("{0} returned {1}", providerName, response.status) +
        (detail ? `: ${detail}` : "")
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
    throw new Error(tx("Enter a valid absolute API endpoint."));
  }
  const localDevelopmentHost = ["localhost", "127.0.0.1", "[::1]"].includes(
    url.hostname
  );
  if (
    url.protocol !== "https:" &&
    !(url.protocol === "http:" && localDevelopmentHost)
  ) {
    throw new Error(
      tx("API endpoints must use HTTPS (HTTP is allowed only on localhost).")
    );
  }
  if (url.username || url.password) {
    throw new Error(tx("Do not place credentials in the API endpoint URL."));
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
