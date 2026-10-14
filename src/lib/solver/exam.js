/**
 * Exam answering for the Multimodal AI Solver: the prompt that asks a model
 * for a structured answer to one exam question (single choice, multiple
 * choice, true/false, fill-in or short answer), parsing of that answer, and
 * comparison of two independent answers.
 */

export const QUESTION_TYPES = [
  "single",
  "multiple",
  "true_false",
  "fill",
  "short",
];

export function examSystemPrompt(chinese) {
  return [
    "You answer laboratory safety exam questions (Chinese university and institute lab safety training and access exams) accurately.",
    "Base answers on established laboratory safety practice and standards: chemical, biological, radiation, electrical, fire, pressure-vessel and gas-cylinder safety, personal protection, waste disposal and emergency response.",
    "Read every option. Where the exam wording follows a common textbook rule, give that rule's answer.",
    "Return only a JSON object, no other text.",
    chinese
      ? "Write question, reason and basis in Simplified Chinese."
      : "Write question, reason and basis in the language of the question.",
  ].join(" ");
}

/**
 * The user prompt. `variant` "check" asks for an independent second answer
 * that weighs each option first, so the two answers are not the same text.
 */
export function examPrompt({ text = "", variant = "main" } = {}) {
  const source = text.trim()
    ? `Question:\n${text.trim()}`
    : "The question is in the image. Read it exactly, including every option.";
  const method =
    variant === "check"
      ? "First judge each option (or the statement) as right or wrong on its own, then decide the answer."
      : "Decide the answer directly from the rule that applies.";
  return `${source}

${method}

Return JSON:
{
  "question": "the question and its options as read",
  "question_type": "single | multiple | true_false | fill | short",
  "answer": "option letters such as \\"B\\" or \\"ACD\\"; \\"true\\" or \\"false\\" for true/false; the text for fill-in or short answer",
  "reason": "one or two sentences explaining why",
  "basis": "the safety rule or standard that applies, if any",
  "confidence": "high | medium | low"
}
For several questions in one image, answer only the first complete one and say so in "reason".`;
}

function stripFence(text) {
  return String(text || "")
    .trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/```\s*$/, "")
    .trim();
}

function parseJsonLoose(text) {
  const clean = stripFence(text);
  try {
    return JSON.parse(clean);
  } catch {
    const start = clean.indexOf("{");
    const end = clean.lastIndexOf("}");
    if (start !== -1 && end > start) {
      try {
        return JSON.parse(clean.slice(start, end + 1));
      } catch {
        return null;
      }
    }
    return null;
  }
}

const TRUE_WORDS = /^(true|yes|correct|right|对|正确|是|√|✓|✔)$/i;
const FALSE_WORDS = /^(false|no|incorrect|wrong|错|错误|否|×|✗|✘)$/i;

/**
 * Normalized answer: { kind: "letters", letters: "ACD" } | { kind: "bool",
 * value } | { kind: "text", text }.
 */
export function normalizeAnswer(answer, type) {
  if (typeof answer === "boolean") return { kind: "bool", value: answer };
  const raw = Array.isArray(answer)
    ? answer.join("")
    : String(answer ?? "").trim();
  const compact = raw.replace(/[\s,，、;；.。:：()（）[\]"'“”]/g, "");
  if (TRUE_WORDS.test(compact)) return { kind: "bool", value: true };
  if (FALSE_WORDS.test(compact)) return { kind: "bool", value: false };
  if (type === "true_false") {
    if (/^(对|正确|true)/i.test(compact)) return { kind: "bool", value: true };
    if (/^(错|错误|false)/i.test(compact)) {
      return { kind: "bool", value: false };
    }
  }
  if (/^[A-Ha-h]+$/.test(compact)) {
    const letters = [...new Set(compact.toUpperCase())].sort().join("");
    return { kind: "letters", letters };
  }
  // "选B" / "答案为 A、C"
  const lettersOnly = raw.match(/\b[A-H]\b/g);
  if (
    lettersOnly &&
    (type === "single" || type === "multiple") &&
    raw.replace(/[^A-Za-z]/g, "").length === lettersOnly.length
  ) {
    return {
      kind: "letters",
      letters: [...new Set(lettersOnly)].sort().join(""),
    };
  }
  return { kind: "text", text: raw };
}

/** Parse a model reply into an exam answer, or null if it has none. */
export function parseExamAnswer(text) {
  const json = parseJsonLoose(text);
  if (json && typeof json === "object" && json.answer !== undefined) {
    const type = QUESTION_TYPES.includes(json.question_type)
      ? json.question_type
      : null;
    const confidence = ["high", "medium", "low"].includes(
      String(json.confidence).toLowerCase()
    )
      ? String(json.confidence).toLowerCase()
      : null;
    return {
      question: typeof json.question === "string" ? json.question : "",
      type,
      answer: normalizeAnswer(json.answer, type),
      reason: typeof json.reason === "string" ? json.reason : "",
      basis: typeof json.basis === "string" ? json.basis : "",
      confidence,
    };
  }
  // Plain-text fallback: a line such as "答案：B" or "Answer: true".
  const line = String(text || "").match(
    /(?:答案|正确答案|answer)\s*[:：为是]?\s*([^\n]+)/i
  );
  if (!line) return null;
  return {
    question: "",
    type: null,
    answer: normalizeAnswer(line[1].trim(), null),
    reason: "",
    basis: "",
    confidence: null,
  };
}

/** Whether two normalized answers agree. */
export function sameAnswer(a, b) {
  if (!a || !b || a.kind !== b.kind) return false;
  if (a.kind === "letters") return a.letters === b.letters;
  if (a.kind === "bool") return a.value === b.value;
  const norm = (s) => s.replace(/[\s,，、;；.。:：]/g, "").toLowerCase();
  return norm(a.text) === norm(b.text);
}

/** Display text of a normalized answer. */
export function answerLabel(answer, { yes = "True", no = "False" } = {}) {
  if (!answer) return "";
  if (answer.kind === "letters") return answer.letters;
  if (answer.kind === "bool") return answer.value ? yes : no;
  return answer.text;
}

/** Whether an answer matches a key ("B", "ACD", true, false). */
export function matchesKey(answer, key) {
  return sameAnswer(answer, normalizeAnswer(key, null));
}
