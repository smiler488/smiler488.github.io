/**
 * Datasets (design/DESIGN_SPEC.md §4.3): the data behind the notes and
 * projects. /data, project evidence bars, JSON-LD and llms.txt read from
 * here. `layers` uses the four-layer IDs and is never translated.
 *
 * access: "download" — files are served from static/data on this site;
 *         "request"  — available from the author, as stated in the paper.
 * `license`: the author's choice (2026-11): CC BY-NC 4.0 for the files served
 * here; data available on request keep the terms stated in the paper.
 * `doi` stays null until a dataset is archived (for example on Zenodo).
 */

export const LICENSES = {
  "CC-BY-NC-4.0": {
    name: "CC BY-NC 4.0",
    url: "https://creativecommons.org/licenses/by-nc/4.0/",
  },
};
import { COTTON3D } from "./cotton3d";

const WEB_POINTS_FORMAT = {
  en: "Binary, little-endian: Uint16 x, y, z for every point (quantized over the bounding box below), then one Uint8 organ class per point. Classes: 0 main stem, 1 branches and petioles, 2 leaves.",
  zh: "二进制（小端序）：每个点先存 Uint16 的 x、y、z（在下方包围盒内量化），随后每个点一个 Uint8 器官类别。类别：0 主茎，1 分枝与叶柄，2 叶片。",
};

export const DATASETS = [
  {
    id: "cotton-sfm-pointcloud",
    title: {
      en: "Potted cotton point cloud from turntable photogrammetry (SfM)",
      zh: "转台摄影测量（SfM）重建的盆栽棉花点云",
    },
    summary: {
      en: "One growth-chamber cotton plant (sample cotton_20240109-84-5) reconstructed by structure from motion and labelled by organ.",
      zh: "一株生长室盆栽棉花（样本 cotton_20240109-84-5），以运动恢复结构（SfM）重建，并按器官标注。",
    },
    layers: ["DIG"],
    year: 2024,
    records: {
      en: `${COTTON3D.sfm.count.toLocaleString("en")} points · 3 organ classes`,
      zh: `${COTTON3D.sfm.count.toLocaleString("en")} 个点 · 3 类器官`,
    },
    access: "download",
    files: [{ href: "/data/cotton3d/sfm.bin", bytes: COTTON3D.sfm.bytes }],
    format: WEB_POINTS_FORMAT,
    bounds: { min: COTTON3D.sfm.min, max: COTTON3D.sfm.max, unit: "m" },
    note: "/blog/growth-chamber-cotton-3d",
    doi: null,
    license: "CC-BY-NC-4.0",
  },
  {
    id: "cotton-generative-pointcloud",
    title: {
      en: "Cotton point cloud from single-image 3D generative AI",
      zh: "单图像三维生成式人工智能重建的棉花点云",
    },
    summary: {
      en: "The same plant reconstructed from a single image with 3D generative AI, for comparison against the SfM reference.",
      zh: "同一株棉花由单张图像经三维生成式人工智能重建，用于与 SfM 参考点云对比。",
    },
    layers: ["DIG"],
    year: 2025,
    records: {
      en: `${COTTON3D.gen.count.toLocaleString("en")} points (voxel-downsampled from ${COTTON3D.gen.sourceCount.toLocaleString("en")}) · 3 organ classes`,
      zh: `${COTTON3D.gen.count.toLocaleString("en")} 个点（由 ${COTTON3D.gen.sourceCount.toLocaleString("en")} 个点按类别体素降采样）· 3 类器官`,
    },
    access: "download",
    files: [{ href: "/data/cotton3d/gen.bin", bytes: COTTON3D.gen.bytes }],
    format: WEB_POINTS_FORMAT,
    bounds: { min: COTTON3D.gen.min, max: COTTON3D.gen.max, unit: "m" },
    note: "/blog/generative-ai-3d-plant-reconstruction",
    doi: null,
    license: "CC-BY-NC-4.0",
  },
  {
    id: "cotton-canopy-mesh",
    title: {
      en: "Cotton canopy triangle mesh with leaf optical properties",
      zh: "带叶片光学属性的棉花冠层三角面元模型",
    },
    summary: {
      en: "A 24-plant canopy built from one reconstructed plant, with reflectance and transmittance per optical class, as used for canopy light and photosynthesis modelling.",
      zh: "由同一株重建植株复制排布而成的 24 株冠层，按光学类别赋反射率与透射率，用于冠层光分布与光合建模。",
    },
    layers: ["UND", "PRE"],
    year: 2023,
    records: {
      en: `${COTTON3D.canopy.full.triangles.toLocaleString("en")} triangles in the full model (${(COTTON3D.canopy.leaf.triangles + COTTON3D.canopy.nonleaf.triangles).toLocaleString("en")} in the web file) · leaf area ${COTTON3D.canopy.full.leafArea_m2} m² · height ${COTTON3D.canopy.full.height_m} m`,
      zh: `完整模型 ${COTTON3D.canopy.full.triangles.toLocaleString("en")} 个三角面元（网页文件 ${(COTTON3D.canopy.leaf.triangles + COTTON3D.canopy.nonleaf.triangles).toLocaleString("en")} 个）· 叶面积 ${COTTON3D.canopy.full.leafArea_m2} m² · 高度 ${COTTON3D.canopy.full.height_m} m`,
    },
    access: "download",
    files: [{ href: "/data/cotton3d/canopy.bin", bytes: 1176666 }],
    format: {
      en: `Binary, little-endian: Uint16 x, y, z for the leaf vertices (${COTTON3D.canopy.leaf.vertices.toLocaleString("en")}) then the non-leaf vertices (${COTTON3D.canopy.nonleaf.vertices.toLocaleString("en")}), quantized over the bounding box; then Uint16 index triples for the leaf triangles, then the non-leaf triangles (indices per group). Optical properties: leaf reflectance ${COTTON3D.canopy.full.optical.leaf.reflectance}, transmittance ${COTTON3D.canopy.full.optical.leaf.transmittance}; other organs ${COTTON3D.canopy.full.optical.nonleaf.reflectance} and ${COTTON3D.canopy.full.optical.nonleaf.transmittance}. The web file is simplified per optical class by quadric decimation.`,
      zh: `二进制（小端序）：先存叶片顶点（${COTTON3D.canopy.leaf.vertices.toLocaleString("en")} 个），再存非叶顶点（${COTTON3D.canopy.nonleaf.vertices.toLocaleString("en")} 个），均为包围盒内量化的 Uint16 x、y、z；随后是叶片三角形、非叶三角形的 Uint16 索引三元组（各组内编号）。光学属性：叶片反射率 ${COTTON3D.canopy.full.optical.leaf.reflectance}、透射率 ${COTTON3D.canopy.full.optical.leaf.transmittance}；其他器官 ${COTTON3D.canopy.full.optical.nonleaf.reflectance} 与 ${COTTON3D.canopy.full.optical.nonleaf.transmittance}。网页文件按光学类别做了二次误差简化。`,
    },
    bounds: { min: COTTON3D.canopy.min, max: COTTON3D.canopy.max, unit: "m" },
    note: "/blog/canopy-photosynthesis-modeling-en",
    doi: null,
    license: "CC-BY-NC-4.0",
  },
  {
    id: "leaf-brdf-traits",
    title: {
      en: "Leaf directional spectra, BRDF parameters and phenotypic traits",
      zh: "叶片方向光谱、BRDF 参数与表型性状",
    },
    summary: {
      en: "Directional reflectance (400–1000 nm) of maize, rice, cotton and poplar leaves with fitted Cook–Torrance parameters and the measured traits used to predict them.",
      zh: "玉米、水稻、棉花和杨树叶片的方向反射光谱（400–1000 nm）、拟合的 Cook–Torrance 参数，以及用于预测这些参数的表型性状。",
    },
    layers: ["UND"],
    year: 2025,
    records: {
      en: "270 entries · 4 species · 2 canopy positions · both leaf surfaces",
      zh: "270 条数据 · 4 个物种 · 2 个冠层位置 · 叶片正反两面",
    },
    access: "request",
    files: [],
    publication: "deng2025brdf",
    project: "/research/brdf-traits",
    code: "https://github.com/PlantSystemsBiology/brdf",
    doi: null,
    license: null,
  },
];

export function getDataset(id) {
  return DATASETS.find((dataset) => dataset.id === id);
}
