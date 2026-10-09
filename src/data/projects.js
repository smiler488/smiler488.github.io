/**
 * Research projects (design/DESIGN_SPEC.md §4.3, §6.3). Each project has an
 * MDX page at src/pages/research/<id>.mdx (zh copy under
 * i18n/zh-Hans/docusaurus-plugin-content-pages/research/) whose front matter
 * sets `project: <id>`; the page layout reads the header data from here.
 *
 * Content comes from the author's published papers and notes. `finding` is a
 * result, written as a sentence, not a topic.
 */

export const PROJECT_STATUS = {
  published: { en: "Published", zh: "已发表" },
  active: { en: "Active", zh: "进行中" },
  snapshot: { en: "Project snapshot", zh: "项目快照" },
  archived: { en: "Archived", zh: "已归档" },
};

export const PROJECTS = [
  {
    id: "brdf-traits",
    title: {
      en: "Predicting leaf BRDF from phenotypic traits",
      zh: "从表型性状预测叶片 BRDF",
    },
    finding: {
      en: "Leaf directional reflectance can be predicted from measurable traits, and the optical diversity this reveals changes how light is distributed inside a simulated canopy.",
      zh: "叶片的方向反射可以由可测量的表型性状预测，而由此揭示的叶片光学多样性会改变模拟冠层内部的光分布。",
    },
    layers: ["UND"],
    status: "published",
    year: 2025,
    cover: "/img/brdf_cover.jpg",
    coverAlt: {
      en: "Directional spectrum measurement and BRDF prediction workflow",
      zh: "方向光谱测量与 BRDF 预测流程",
    },
    publications: ["deng2025brdf"],
    code: "https://github.com/PlantSystemsBiology/brdf",
    note: "/blog/brdf-paper",
  },
  {
    id: "mctp-workspace",
    title: {
      en: "MCTP: a multi-modal crop phenotyping workspace",
      zh: "MCTP：多模态作物表型工作台",
    },
    finding: {
      en: "One desktop workspace gives hyperspectral, LiDAR, RGB and thermal processing a shared entry point and export convention, while keeping each modality's processing transparent and tunable.",
      zh: "一个桌面工作台为高光谱、激光雷达、RGB 与热红外处理提供统一入口和一致的导出约定，同时让每种模态的处理过程保持透明、可调。",
    },
    layers: ["DIG"],
    status: "snapshot",
    year: 2025,
    cover: "/img/mctp.png",
    coverAlt: {
      en: "MCTP desktop launcher with four modality modules",
      zh: "MCTP 桌面启动器与四个模态模块",
    },
    publications: [],
    note: "/blog/mctp-unified-phenotyping-platform",
  },
];

export function getProject(id) {
  return PROJECTS.find((project) => project.id === id);
}
