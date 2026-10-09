/**
 * Standard data types exchanged between App Lab tools
 * (design/DESIGN_SPEC.md §10.3). Tools declare them in appManifest.js as
 * `inputs` / `outputs`; matching types drive the "next step" suggestions.
 */
export const DATA_TYPES = {
  "geo.point": { en: "Location", zh: "位置" },
  "geo.point[]": { en: "GPS points", zh: "GPS 点集" },
  "geo.polygon": { en: "Field boundary", zh: "田块边界" },
  "table.timeseries": { en: "Time series table", zh: "时间序列表" },
  table: { en: "Data table", zh: "数据表" },
  image: { en: "Image", zh: "图像" },
  "image.set": { en: "Image set", zh: "图像集" },
  document: { en: "Document", zh: "文档" },
  text: { en: "Text", zh: "文本" },
};

export const MATURITY = {
  stable: {
    en: "Stable",
    zh: "稳定",
    hint: {
      en: "Method published or validated; results can be used in research.",
      zh: "方法已发表或经过验证，结果可用于科研。",
    },
  },
  beta: {
    en: "Beta",
    zh: "测试中",
    hint: {
      en: "Feature-complete; validation is ongoing.",
      zh: "功能完整，验证进行中。",
    },
  },
  experimental: {
    en: "Experimental",
    zh: "实验性",
    hint: {
      en: "Exploratory; treat results as indicative only.",
      zh: "探索性功能，结果仅供参考。",
    },
  },
};

/** Mean of the vertices of a lat/lng ring: a representative field location. */
export function polygonCentroid(points) {
  if (!points?.length) return null;
  const lat = points.reduce((s, p) => s + p.lat, 0) / points.length;
  const lng = points.reduce((s, p) => s + p.lng, 0) / points.length;
  return { lat, lng };
}
