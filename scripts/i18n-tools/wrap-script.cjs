/**
 * Static App Lab scripts (static/js/<tool>_app.js): wraps status messages,
 * alerts, progress text and textContent assignments in tx<Tool>("…") and
 * writes the dictionary to static/js/i18n/<tool>.zh.js
 * (window.__ZH_<TOOL> = {…}). The page loads that file before the script.
 *
 * Usage: node scripts/i18n-tools/wrap-script.cjs <tool> <message functions…>
 */
const fs = require("fs");
const parser = require("@babel/parser");
const traverse = require("@babel/traverse").default;
const { isTextString, templateKey } = require("./rules.cjs");

const [tool, ...fns] = process.argv.slice(2);
const Tool = tool[0].toUpperCase() + tool.slice(1);
const fnName = `tx${Tool}`;
const globalName = `__ZH_${tool.toUpperCase()}`;
const file = `static/js/${tool}_app.js`;
const code = fs.readFileSync(file, "utf8");
const ast = parser.parse(code, { sourceType: "script" });
const VARS = new Set((fns.find((f) => f.startsWith("vars=")) || "vars=").slice(5).split(",").filter(Boolean));
fns.splice(0, fns.length, ...fns.filter((f) => !f.startsWith("vars=")));
const ARG = Object.fromEntries(fns.map((f) => (f.includes(":") ? f.split(":") : [f, "0"])).map(([f, i]) => [f, Number(i)]));
const edits = [];
const keys = [];

function wrap(p) {
  if (!p || Array.isArray(p) || !p.node) return;
  const n = p.node;
  if (!n) return;
  if (n.type === "StringLiteral" && isTextString(n.value)) {
    edits.push({ s: n.start, e: n.end, t: `${fnName}(${JSON.stringify(n.value)})` });
    keys.push(n.value.trim());
  } else if (n.type === "TemplateLiteral" && isTextString(templateKey(n))) {
    const key = templateKey(n);
    const args = n.expressions.map((x) => code.slice(x.start, x.end));
    edits.push({ s: n.start, e: n.end, t: `${fnName}(${[JSON.stringify(key), ...args].join(", ")})` });
    keys.push(key.trim());
  } else if (n.type === "ConditionalExpression") {
    wrap(p.get("consequent"));
    wrap(p.get("alternate"));
  } else if (n.type === "LogicalExpression") wrap(p.get("right"));
}
const inTx = (p) => { for (let q = p.parentPath; q; q = q.parentPath) if (q.node.type === "CallExpression" && q.node.callee.name === fnName) return true; return false; };

traverse(ast, {
  CallExpression(p) {
    const c = p.node.callee;
    const name = c.type === "Identifier" ? c.name : c.type === "MemberExpression" && !c.computed && c.object.name !== "console" ? c.property.name : null;
    if (name && Object.hasOwn(ARG, name) && !inTx(p)) wrap(p.get(`arguments.${ARG[name]}`));
  },
  NewExpression(p) {
    if (p.node.callee.name === "Error" && !inTx(p)) wrap(p.get("arguments.0"));
  },
  VariableDeclarator(p) {
    if (p.node.id.type === "Identifier" && VARS.has(p.node.id.name) && !inTx(p)) wrap(p.get("init"));
  },
  AssignmentExpression(p) {
    const l = p.node.left;
    if (l.type === "MemberExpression" && ["textContent", "innerText"].includes(l.property.name) && !inTx(p)) wrap(p.get("right"));
  },
});

let out = code;
const seen = new Set();
for (const e of edits.sort((a, b) => b.s - a.s)) {
  if (seen.has(e.s)) continue;
  seen.add(e.s);
  out = out.slice(0, e.s) + e.t + out.slice(e.e);
}
if (!out.includes(`function ${fnName}(`)) {
  out = `/* Interface text: English here, Chinese in static/js/i18n/${tool}.zh.js. */
function ${fnName}(text) {
  var args = Array.prototype.slice.call(arguments, 1);
  var dict = (typeof window !== "undefined" && window.${globalName}) || {};
  var zh = typeof document !== "undefined" && document.documentElement.lang === "zh-Hans";
  var m = String(text).match(/^(\\s*)([\\s\\S]*?)(\\s*)$/);
  var core = zh && dict[m[2]] ? dict[m[2]] : m[2];
  return (m[1] + core + m[3]).replace(/\\{(\\d+)\\}/g, function (s, i) {
    return i < args.length ? String(args[i]) : s;
  });
}

` + out;
}
fs.writeFileSync(file, out);

fs.mkdirSync("static/js/i18n", { recursive: true });
const dictFile = `static/js/i18n/${tool}.zh.js`;
let dict = {};
if (fs.existsSync(dictFile)) dict = new Function(fs.readFileSync(dictFile, "utf8").replace(/^[\s\S]*?window\.\w+ =/, "return"))();
const all = [...new Set([...Object.keys(dict), ...keys])];
fs.writeFileSync(dictFile, `/** Chinese interface text for ${file}. */\nwindow.${globalName} = {\n${all.map((k) => `  ${JSON.stringify(k)}: ${JSON.stringify(dict[k] ?? "")},`).join("\n")}\n};\n`);
console.log(`${file}: ${edits.length} wrapped, ${all.length} keys`);
