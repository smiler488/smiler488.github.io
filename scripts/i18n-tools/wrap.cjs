/**
 * Wraps user-facing English strings of an App Lab React file in tx("…") and
 * writes/updates the Chinese dictionary next to it (zh.js). Idempotent:
 * strings already inside tx() are skipped; existing translations are kept.
 *
 * Usage: node scripts/i18n-tools/wrap.cjs <file>...
 */
const fs = require("fs");
const path = require("path");
const { collect } = require("./rules.cjs");

function inside(p, test) {
  for (let q = p.parentPath; q; q = q.parentPath) if (test(q)) return true;
  return false;
}
const isTxCall = (q) =>
  q.node.type === "CallExpression" && q.node.callee.type === "Identifier" && q.node.callee.name === "tx";
const isLocaleBranch = (q) =>
  q.node.type === "ObjectProperty" &&
  ["en", "zh"].includes(q.node.key.name ?? q.node.key.value);
const isConsole = (q) =>
  q.node.type === "CallExpression" &&
  q.node.callee.type === "MemberExpression" &&
  q.node.callee.object.type === "Identifier" &&
  q.node.callee.object.name === "console";

function readDict(file) {
  if (!fs.existsSync(file)) return {};
  const src = fs.readFileSync(file, "utf8").replace(/^[\s\S]*?export default/, "return");
  return new Function(src)();
}

for (const file of process.argv.slice(2)) {
  const code = fs.readFileSync(file, "utf8");
  const items = collect(code).filter(
    (i) => !inside(i.path, isTxCall) && !inside(i.path, isLocaleBranch) && !inside(i.path, isConsole)
  );
  const edits = new Map();
  for (const i of items) {
    const n = i.path.node;
    let text;
    if (i.kind === "jsxtext") text = `{tx(${JSON.stringify(i.cleaned)})}`;
    else if (i.kind === "attr") text = `{tx(${JSON.stringify(n.value.value)})}`;
    else if (i.kind === "str") text = `tx(${JSON.stringify(n.value)})`;
    else if (i.kind === "tpl") {
      let key = "";
      n.quasis.forEach((q, k) => {
        key += q.value.cooked;
        if (k < n.expressions.length) key += `{${k}}`;
      });
      const args = n.expressions.map((e) => code.slice(e.start, e.end));
      text = `tx(${[JSON.stringify(key), ...args].join(", ")})`;
    }
    const start = i.kind === "attr" ? n.value.start : n.start;
    const end = i.kind === "attr" ? n.value.end : n.end;
    if (!edits.has(start)) edits.set(start, { start, end, text, key: i.key });
  }
  // Drop edits nested inside another edit (outer wins).
  const list = [...edits.values()].sort((a, b) => a.start - b.start);
  const kept = list.filter((e) => !list.some((o) => o !== e && o.start <= e.start && o.end >= e.end && (o.start !== e.start || o.end !== e.end)));
  let out = code;
  for (const e of kept.sort((a, b) => b.start - a.start)) out = out.slice(0, e.start) + e.text + out.slice(e.end);

  if (kept.length && !/makeToolText/.test(out)) {
    const lines = out.split("\n");
    let last = -1;
    lines.forEach((l, idx) => { if (/^import .* from |^} from /.test(l)) last = idx; });
    lines.splice(last + 1, 0,
      'import { makeToolText } from "@site/src/lib/i18n/toolText";',
      `import ZH from "./${file.includes("src/pages/") ? "_zh" : "zh"}";`,
      "",
      "const tx = makeToolText(ZH);");
    out = lines.join("\n");
  }
  fs.writeFileSync(file, out);

  const dictFile = path.join(path.dirname(file), file.includes("src/pages/") ? "_zh.js" : "zh.js");
  const dict = readDict(dictFile);
  const keys = [...new Set([...Object.keys(dict), ...kept.map((e) => e.key)])];
  const body = keys.map((k) => `  ${JSON.stringify(k)}: ${JSON.stringify(dict[k] ?? "")},`).join("\n");
  fs.writeFileSync(
    dictFile,
    `/** Chinese interface text for this tool (src/lib/i18n/toolText.js). */\nexport default {\n${body}\n};\n`
  );
  console.log(`${file}: ${kept.length} wrapped, ${keys.length} keys`);
}
