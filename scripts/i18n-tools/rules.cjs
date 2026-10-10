/**
 * Which strings in a React file are user-facing text. Shared by extract.cjs
 * (listing) and wrap.cjs (rewriting to t("…")).
 */
const parser = require("@babel/parser");
const traverse = require("@babel/traverse").default;

const TEXT_ATTRS = new Set([
  "label", "placeholder", "title", "aria-label", "alt", "description",
  "hint", "suffix", "summary", "caption", "aria-description", "emptyText",
]);
const MESSAGE_CALLS = new Set([
  "setStatus", "setError", "updateStatus", "setStatusMessage", "setChartError",
  "setChartRuntimeError", "alert", "setMessage", "setNotice", "setWarning",
  "setInfo", "setToast", "showStatus", "setLoadingMessage", "setAiSummary",
  "setHint", "setFeedback", "setResult", "setErrorMessage", "setStatusText",
  "setNote", "setLog", "log", "setNotification",
]);
const TEXT_KEYS = new Set([
  "label", "description", "hint", "title", "placeholder", "helper", "caption",
  "message", "tooltip", "summary", "help", "note", "subtitle", "detail",
  "text", "desc",
  ...(process.env.EXTRA_KEYS ? process.env.EXTRA_KEYS.split(",") : []),
]);

const hasWords = (s) =>
  (/[A-Za-z]{2,}/.test(s) && !/^[\w.-]+$/.test(s.trim())) ||
  /^[A-Z][a-z]+(\.\.\.|…|:)?$/.test(s.trim());
const looksLikeCode = (s) =>
  /^(https?:|\/|#|\.|[a-z]+\/[a-z]|[a-z_]+\.[a-z]+$|var\(|rgba?\(|\d)/.test(s.trim()) ||
  /^[a-z][a-zA-Z0-9]*$/.test(s.trim()) || // camelCase identifiers
  /[{}<>=]|;\S/.test(s.replace(/\{\d+\}/g, "N"));

// React's JSX whitespace rule (Babel cleanJSXElementLiteralChild).
function cleanJsxText(value) {
  const lines = value.split(/\r\n|\n|\r/);
  let lastNonEmpty = 0;
  lines.forEach((l, i) => { if (/[^ \t]/.test(l)) lastNonEmpty = i; });
  let str = "";
  lines.forEach((line, i) => {
    const isFirst = i === 0;
    const isLast = i === lines.length - 1;
    const isLastNonEmpty = i === lastNonEmpty;
    let trimmed = line.replace(/\t/g, " ");
    if (!isFirst) trimmed = trimmed.replace(/^[ ]+/, "");
    if (!isLast) trimmed = trimmed.replace(/[ ]+$/, "");
    if (trimmed) {
      if (!isLastNonEmpty) trimmed += " ";
      str += trimmed;
    }
  });
  return str;
}

// Template literal → key with {0}, {1} placeholders and the expressions.
function templateKey(node) {
  let key = "";
  node.quasis.forEach((q, i) => {
    key += q.value.cooked;
    if (i < node.expressions.length) key += `{${i}}`;
  });
  return key;
}

function isTextString(s) {
  const core = s.replace(/\{\d+\}/g, "N").trim();
  return core.length > 1 && hasWords(core) && !looksLikeCode(core);
}

function collect(code) {
  const ast = parser.parse(code, { sourceType: "module", plugins: ["jsx"] });
  const items = [];
  const push = (path, kind, key, extra = {}) => items.push({ path, kind, key, ...extra });
  traverse(ast, {
    JSXText(path) {
      const cleaned = cleanJsxText(path.node.value);
      if (isTextString(cleaned)) push(path, "jsxtext", cleaned.trim(), { cleaned });
    },
    JSXAttribute(path) {
      const name = path.node.name.name;
      const v = path.node.value;
      if (!TEXT_ATTRS.has(name) || !v) return;
      if (v.type === "StringLiteral" && isTextString(v.value)) push(path, "attr", v.value.trim());
      if (v.type === "JSXExpressionContainer") {
        const e = v.expression;
        if (e.type === "StringLiteral" && isTextString(e.value)) push(path.get("value.expression"), "str", e.value.trim());
        if (e.type === "TemplateLiteral" && isTextString(templateKey(e))) push(path.get("value.expression"), "tpl", templateKey(e).trim());
      }
    },
    JSXExpressionContainer(path) {
      // {"Text"}, {cond ? "A" : "B"}, {x || "Fallback"} as element children.
      if (path.parent.type === "JSXAttribute") return;
      const e = path.node.expression;
      if (["StringLiteral", "TemplateLiteral", "ConditionalExpression", "LogicalExpression"].includes(e.type))
        visitValue(path.get("expression"));
    },
    VariableDeclarator(path) {
      // Module-level lists of sentences: const tips = ["…", "…"].
      const init = path.node.init;
      if (!init || init.type !== "ArrayExpression") return;
      path.get("init.elements").forEach((el) => {
        if (el.node && el.node.type === "StringLiteral" && el.node.value.split(" ").length >= 4) visitValue(el);
      });
    },
    CallExpression(path) {
      const c = path.node.callee;
      const name = c.type === "Identifier" ? c.name : c.type === "MemberExpression" && c.property.type === "Identifier" ? c.property.name : null;
      if (!MESSAGE_CALLS.has(name)) return;
      const arg = path.get("arguments.0");
      if (!arg || !arg.node) return;
      visitValue(arg);
    },
    NewExpression(path) {
      if (path.node.callee.type === "Identifier" && path.node.callee.name === "Error") {
        const arg = path.get("arguments.0");
        if (arg && arg.node) visitValue(arg);
      }
    },
    ObjectProperty(path) {
      const k = path.node.key;
      const name = k.type === "Identifier" ? k.name : k.type === "StringLiteral" ? k.value : null;
      if (!TEXT_KEYS.has(name)) return;
      // Skip bilingual objects ({ en, zh }) that are already localized.
      visitValue(path.get("value"));
    },
  });
  function visitValue(p) {
    const n = p.node;
    if (n.type === "StringLiteral" && isTextString(n.value)) push(p, "str", n.value.trim());
    else if (n.type === "TemplateLiteral" && isTextString(templateKey(n))) push(p, "tpl", templateKey(n).trim());
    else if (n.type === "ConditionalExpression") {
      visitValue(p.get("consequent"));
      visitValue(p.get("alternate"));
    } else if (n.type === "LogicalExpression") visitValue(p.get("right"));
  }
  return items;
}

module.exports = { collect, cleanJsxText, templateKey, isTextString };
