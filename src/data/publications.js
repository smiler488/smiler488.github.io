/**
 * Single source of truth for publications and citable research outputs
 * (design/DESIGN_SPEC.md §4.3). The CV, /publications, project pages,
 * structured data and llms.txt all read from here.
 *
 * `layers` uses the four-layer IDs (DIG / UND / PRE / DES) and is never
 * translated. Localized fields are { en, zh }.
 */

export const LAYERS = [
  {
    id: "DIG",
    index: "I",
    name: { en: "Digitize", zh: "数字化" },
    summary: {
      en: "Physical to digital crop: imaging, 3D reconstruction and sensing turn a real crop into quantified traits.",
      zh: "从物理作物到数字作物：成像、三维重建与传感，把真实作物变成可量化的性状。",
    },
  },
  {
    id: "UND",
    index: "II",
    name: { en: "Understand", zh: "理解" },
    summary: {
      en: "Digital to explainable crop: coupling structure, radiation and photosynthesis with scientific AI.",
      zh: "从数字作物到可解释作物：耦合结构、辐射与光合过程和科学 AI。",
    },
  },
  {
    id: "PRE",
    index: "III",
    name: { en: "Predict", zh: "预测" },
    summary: {
      en: "Toward a predictive crop: projecting growth under environment and management scenarios.",
      zh: "走向可预测作物：在环境与管理情景下预测生长，并量化决策风险。",
    },
  },
  {
    id: "DES",
    index: "IV",
    name: { en: "Design", zh: "设计" },
    summary: {
      en: "Toward a designed crop: inverse design over genotype, environment and management.",
      zh: "走向可设计作物：在基因型、环境与管理空间中做逆向设计与优化。",
    },
  },
];

export const LAYER_IDS = LAYERS.map((layer) => layer.id);

export function getLayer(id) {
  return LAYERS.find((layer) => layer.id === id);
}

export const PUBLICATIONS = [
  {
    id: "deng2025brdf",
    type: "article",
    year: 2025,
    title:
      "Leaf Bidirectional Reflectance Distribution Function (BRDF) Prediction with Phenotypic Traits in Four Species: Development of a Novel Measuring and Analyzing Framework",
    authors: [
      "Deng, L.",
      "Yu, L. X.",
      "Mao, L.",
      "Wang, Y.",
      "Guo, X.",
      "Wang, M.",
      "Zhang, Y.",
      "Song, Q.",
      "Zhu, X.-G.",
    ],
    venue: "Plant Phenomics",
    volume: "7",
    issue: "4",
    pages: "100135",
    doi: "10.1016/j.plaphe.2025.100135",
    layers: ["UND"],
    project: "brdf-traits",
    code: "https://github.com/PlantSystemsBiology/brdf",
  },
  {
    id: "deng2025platform",
    type: "software",
    year: 2025,
    title: "Digital Plant Phenotyping Platform (v25.0)",
    authors: ["Deng, L."],
    venue: "Zenodo",
    doi: "10.5281/zenodo.17544584",
    layers: ["DIG"],
    description: {
      en: "An integrated platform for plant phenotyping, data processing, and analysis. Core modules have been transferred through Shufeng Bio for applied phenotyping and intelligent-agriculture services.",
      zh: "集成植物表型分析、数据处理与分析的软件平台。核心模块已通过黍峰生物完成转让和商业化，用于植物表型与智慧农业服务。",
    },
  },
];

export const PUBLICATION_KIND = {
  article: { en: "Peer-reviewed article", zh: "同行评议论文", mark: "PP" },
  software: { en: "Research software", zh: "科研软件", mark: "SW" },
};

export function doiUrl(doi) {
  return `https://doi.org/${doi}`;
}

export function getPublication(id) {
  return PUBLICATIONS.find((item) => item.id === id);
}

/** APA-style reference string, used by citation copy buttons. */
export function formatApa(pub) {
  const authors =
    pub.authors.length > 1
      ? `${pub.authors.slice(0, -1).join(", ")}, & ${pub.authors.at(-1)}`
      : pub.authors[0];
  const issue = pub.volume
    ? ` ${pub.volume}${pub.issue ? `(${pub.issue})` : ""}${
        pub.pages ? `, ${pub.pages}` : ""
      }`
    : "";
  return `${authors} (${pub.year}). ${pub.title}. ${
    pub.venue
  },${issue}. ${doiUrl(pub.doi)}`.replace(",.", ".");
}

/** BibTeX entry for a publication. */
export function formatBibtex(pub) {
  const type = pub.type === "software" ? "software" : "article";
  const fields = [
    ["title", pub.title],
    ["author", pub.authors.join(" and ")],
    [pub.type === "software" ? "publisher" : "journal", pub.venue],
    ["year", String(pub.year)],
    ["volume", pub.volume],
    ["number", pub.issue],
    ["pages", pub.pages],
    ["doi", pub.doi],
  ].filter(([, value]) => value);
  return `@${type}{${pub.id},\n${fields
    .map(([key, value]) => `  ${key} = {${value}}`)
    .join(",\n")}\n}`;
}
