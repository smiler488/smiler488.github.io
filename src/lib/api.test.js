/**
 * Request shapes for every API format the AI tools speak, and consistency of
 * the provider presets (src/lib/aiProviders.js).
 *
 * Run: npm run test:science
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  AI_PROVIDER_GROUPS,
  AI_PROVIDER_PRESETS,
  buildAIRequest,
  changeAIProvider,
  createDefaultAIConfig,
  extractAssistantText,
  requestAI,
} from "./api.js";

const PROTOCOLS = new Set([
  "mock",
  "openai",
  "openai-responses",
  "anthropic",
  "gemini",
]);
const IMAGE = "data:image/png;base64,iVBORw0KGgo=";

function configFor(providerId, extra = {}) {
  return {
    ...changeAIProvider(createDefaultAIConfig(), providerId),
    apiKey: "test-key",
    ...extra,
  };
}

test("presets are well formed", () => {
  const ids = new Set();
  const groups = new Set(AI_PROVIDER_GROUPS.map((group) => group.id));
  for (const p of AI_PROVIDER_PRESETS) {
    assert.ok(!ids.has(p.id), `duplicate provider ${p.id}`);
    ids.add(p.id);
    assert.ok(groups.has(p.group), `${p.id} group`);
    assert.ok(PROTOCOLS.has(p.protocol), `${p.id} protocol`);
    assert.ok(p.name.en && p.name.zh, `${p.id} name`);
    assert.ok(p.description.en && p.description.zh, `${p.id} description`);
    if (p.endpointLocked) {
      assert.match(p.endpoint, /^https:\/\//, `${p.id} endpoint`);
      assert.ok(p.models.length > 0, `${p.id} models`);
    }
    const modelIds = p.models.map((m) => m.id);
    assert.equal(new Set(modelIds).size, modelIds.length, `${p.id} models`);
  }
  for (const id of ["demo", "openai", "anthropic", "gemini", "custom"]) {
    assert.ok(ids.has(id), `missing ${id}`);
  }
});

test("OpenAI-compatible chat completions", () => {
  const req = buildAIRequest(
    configFor("deepseek"),
    { question: "Hi", imageBase64: IMAGE },
    { jsonMode: true, systemPrompt: "S" }
  );
  assert.equal(req.url, "https://api.deepseek.com/chat/completions");
  assert.equal(req.headers.Authorization, "Bearer test-key");
  assert.equal(req.body.messages[0].content, "S");
  assert.equal(req.body.messages[1].content[1].image_url.url, IMAGE);
  assert.deepEqual(req.body.response_format, { type: "json_object" });
});

test("OpenAI Responses API", () => {
  const req = buildAIRequest(
    configFor("openai"),
    { question: "Hi", imageBase64: IMAGE },
    { jsonMode: true, systemPrompt: "S", maxTokens: 900 }
  );
  assert.equal(req.url, "https://api.openai.com/v1/responses");
  assert.equal(req.body.instructions, "S");
  assert.equal(req.body.max_output_tokens, 900);
  assert.deepEqual(req.body.input[0].content, [
    { type: "input_text", text: "Hi" },
    { type: "input_image", image_url: IMAGE },
  ]);
  assert.deepEqual(req.body.text, { format: { type: "json_object" } });
  assert.equal(
    extractAssistantText({
      output: [
        { type: "reasoning", summary: [] },
        {
          type: "message",
          content: [{ type: "output_text", text: '{"ok":true}' }],
        },
      ],
    }),
    '{"ok":true}'
  );
});

test("Anthropic Messages: browser opt-in header only for Anthropic", () => {
  const official = buildAIRequest(configFor("anthropic"), { question: "Hi" });
  assert.equal(official.headers["x-api-key"], "test-key");
  assert.equal(
    official.headers["anthropic-dangerous-direct-browser-access"],
    "true"
  );
  const minimax = buildAIRequest(configFor("minimax"), { question: "Hi" });
  assert.equal(minimax.url, "https://api.minimax.io/anthropic/v1/messages");
  assert.equal(
    minimax.headers["anthropic-dangerous-direct-browser-access"],
    undefined
  );
  assert.equal(
    extractAssistantText({
      content: [
        { type: "thinking", thinking: "…" },
        { type: "text", text: "answer" },
      ],
    }),
    "answer"
  );
});

test("Gemini generateContent", () => {
  const req = buildAIRequest(
    configFor("gemini", { model: "gemini-3.8-flash" }),
    { question: "Hi", imageBase64: IMAGE },
    { jsonMode: true }
  );
  assert.equal(
    req.url,
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
  );
  assert.equal(
    req.body.contents[0].parts[0].inline_data.mime_type,
    "image/png"
  );
  assert.equal(req.body.generationConfig.responseMimeType, "application/json");
});

test("local servers need no key and send no Authorization header", async () => {
  const realFetch = globalThis.fetch;
  let sent;
  globalThis.fetch = async (url, init) => {
    sent = { url, init };
    return {
      ok: true,
      json: async () => ({ choices: [{ message: { content: "local" } }] }),
    };
  };
  try {
    const result = await requestAI(
      { ...configFor("ollama"), apiKey: "", model: "qwen3.6" },
      { question: "Hi" }
    );
    assert.equal(result.text, "local");
    assert.equal(sent.url, "http://localhost:11434/v1/chat/completions");
    assert.equal(sent.init.headers.Authorization, undefined);
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("a blocked request explains itself; vision is checked", async () => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new TypeError("Failed to fetch");
  };
  try {
    await assert.rejects(
      requestAI(configFor("deepseek"), { question: "Hi" }),
      /Could not reach DeepSeek/
    );
    await assert.rejects(
      requestAI(
        configFor("deepseek", { model: "deepseek-v4-pro" }),
        { question: "Hi", imageBase64: IMAGE },
        { requireVision: true }
      ),
      /does not accept image input/
    );
    await assert.rejects(
      requestAI(configFor("openai", { apiKey: "" }), { question: "Hi" }),
      /API key/
    );
  } finally {
    globalThis.fetch = realFetch;
  }
});
