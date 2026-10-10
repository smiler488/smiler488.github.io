/**
 * Lists user-facing English strings in an App Lab React file, using the same
 * rules as wrap.cjs. Usage: node scripts/i18n-tools/extract.cjs <file>...
 */
const fs = require("fs");
const { collect } = require("./rules.cjs");
let total = 0;
for (const file of process.argv.slice(2)) {
  const items = collect(fs.readFileSync(file, "utf8"));
  const uniq = [...new Set(items.map((i) => i.key))];
  total += uniq.length;
  console.log(`${file}: ${uniq.length}`);
  if (process.env.SHOW) uniq.forEach((k) => console.log("   ", JSON.stringify(k)));
}
console.log("total", total);
