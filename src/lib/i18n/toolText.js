/**
 * Interface text for App Lab tools. Each tool keeps its English text in the
 * code and a Chinese dictionary next to it (zh.js); `tx` returns the Chinese
 * text on the zh-Hans site. Keys may hold placeholders {0}, {1} … for values
 * inserted at run time. Edge whitespace of a key is preserved.
 *
 * process.env.SITE_LOCALE is defined per locale build by
 * plugins/site-index (configureWebpack).
 */
export const SITE_LOCALE = process.env.SITE_LOCALE || "en";
export const IS_ZH = SITE_LOCALE === "zh-Hans";

function fill(text, args) {
  return args.length
    ? text.replace(/\{(\d+)\}/g, (m, i) => (i < args.length ? String(args[i]) : m))
    : text;
}

export function makeToolText(dict) {
  return function tx(text, ...args) {
    if (typeof text !== "string") return text;
    if (!IS_ZH) return fill(text, args);
    const m = text.match(/^(\s*)([\s\S]*?)(\s*)$/);
    const core = m[2];
    const zh = (Object.prototype.hasOwnProperty.call(dict, core) && dict[core]) || core;
    return fill(m[1] + zh + m[3], args);
  };
}
