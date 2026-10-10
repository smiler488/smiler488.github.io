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
  archived: { en: "Archived", zh: "已归档" },
};

export const PROJECTS = [
  {
    id: "brdf-traits",
    title: {
      en: "Predicting leaf BRDF from phenotypic traits",
      zh: "基于表型性状预测叶片 BRDF",
    },
    finding: {
      en: "Leaf directional reflectance can be predicted from measurable traits, and the resulting differences in leaf optics change how light is distributed inside a simulated canopy.",
      zh: "叶片的方向反射可以由易测的表型性状预测，叶片光学特性的差异会改变模拟冠层内的光分布。",
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
      en: "MCTP: a multi-modal crop phenotyping data processing platform",
      zh: "MCTP：多模态作物表型数据处理平台",
    },
    finding: {
      en: "One desktop platform processes hyperspectral, LiDAR, RGB and thermal phenotyping data with a shared interface, interactive parameter tuning, batch processing and structured exports.",
      zh: "一个桌面平台以统一界面、交互式调参、批量处理和结构化导出，处理高光谱、LiDAR、RGB 和热红外表型数据。",
    },
    layers: ["DIG"],
    status: "active",
    year: 2025,
    cover: "/img/mctp.png",
    coverAlt: {
      en: "MCTP launcher with four modality modules",
      zh: "MCTP 启动界面与四个模态模块",
    },
    publications: [],
    note: "/blog/mctp-unified-phenotyping-platform",
  },
];

export function getProject(id) {
  return PROJECTS.find((project) => project.id === id);
}
