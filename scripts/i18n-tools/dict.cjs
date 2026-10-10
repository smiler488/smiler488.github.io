/**
 * dump <zh.js>                 → numbered list of keys still untranslated
 * fill <zh.js> <list.json>     → fill those keys, in order, from a JSON array
 */
const fs = require("fs");
const [cmd, file, listFile] = process.argv.slice(2);
const src = fs.readFileSync(file, "utf8");
const HEAD = /^[\s\S]*?(export default|window\.\w+ =)/;
const dict = new Function(src.replace(HEAD, "return"))();
const todo = Object.keys(dict).filter((k) => !dict[k]);
if (cmd === "dump") {
  todo.forEach((k, i) => console.log(`${i + 1}\t${JSON.stringify(k)}`));
} else if (cmd === "fill") {
  const list = JSON.parse(fs.readFileSync(listFile, "utf8"));
  if (list.length !== todo.length) throw new Error(`${file}: ${todo.length} keys, ${list.length} translations`);
  todo.forEach((k, i) => (dict[k] = list[i]));
  const body = Object.keys(dict).map((k) => `  ${JSON.stringify(k)}: ${JSON.stringify(dict[k])},`).join("\n");
  const head = src.match(HEAD)[0];
  fs.writeFileSync(file, `${head} {\n${body}\n};\n`);
  console.log(`${file}: filled ${list.length}`);
}
