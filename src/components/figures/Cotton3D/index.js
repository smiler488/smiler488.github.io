/**
 * Interactive 3D figures for the cotton reconstruction notes
 * (design/DESIGN_SPEC.md §7.1–7.2). Each shows a static poster first and
 * fetches three.js plus the data only when the figure scrolls into view.
 */
import React from "react";
import useBaseUrl from "@docusaurus/useBaseUrl";
import { InteractiveFigure } from "@site/src/components/figure";
import { useIsChinese } from "@site/src/components/ds";
import { COTTON3D } from "@site/src/data/cotton3d";

const loadViewer = () => import("./Viewer");

const fmt = (n, zh) => n.toLocaleString(zh ? "zh-CN" : "en");

const COPY = {
  en: {
    sfmTitle: "SfM point cloud of a cotton plant",
    sfmCaption: (n) =>
      `Sample cotton_20240109-84-5, reconstructed with structure from motion: all ${n} points, coloured by organ class.`,
    compareTitle: "SfM and Hunyuan3D reconstructions of the same plant",
    compareCaption: (sfm, hy, hySrc) =>
      `Sample cotton_20240109-84-5. The Hunyuan3D point cloud is aligned to the SfM reconstruction, so Overlay shows where the generated geometry departs from the measured one. SfM: ${sfm} points. Hunyuan3D: ${hySrc} points, downsampled within each organ class to ${hy} for the web.`,
    canopyTitle: "Triangle-facet canopy model for light simulation",
    canopyCaption: (full, shown, leafArea, nonArea) =>
      `24 cotton plants built from single-plant reconstructions, as used in the canopy photosynthesis model. The full model has ${full} triangles (one-sided leaf area ${leafArea} m², other organs ${nonArea} m²); ${shown} are shown here after simplification for the web, so the view is for inspection, not measurement.`,
    posterAlt: "Static view of the same 3D data",
  },
  zh: {
    sfmTitle: "棉花单株的 SfM 点云",
    sfmCaption: (n) =>
      `样本 cotton_20240109-84-5，运动恢复结构（SfM）重建，共 ${n} 个点，按器官类别着色。`,
    compareTitle: "同一植株的 SfM 与 Hunyuan3D 重建",
    compareCaption: (sfm, hy, hySrc) =>
      `样本 cotton_20240109-84-5。Hunyuan3D 点云已与 SfM 重建对齐，“叠加”视图可直接看出生成几何与实测几何的偏差。SfM：${sfm} 个点；Hunyuan3D：${hySrc} 个点，为网页显示在各器官类别内降采样至 ${hy} 个。`,
    canopyTitle: "用于光照模拟的三角面元冠层模型",
    canopyCaption: (full, shown, leafArea, nonArea) =>
      `由单株重建构建的 24 株棉花冠层，即冠层光合模型所用的输入。完整模型共 ${full} 个三角面元（叶片单面面积 ${leafArea} m²，其他器官 ${nonArea} m²）；此处为网页显示简化为 ${shown} 个，仅供查看，不用于测量。`,
    posterAlt: "同一三维数据的静态视图",
  },
};

function useCopy() {
  const zh = useIsChinese();
  return [zh ? COPY.zh : COPY.en, zh];
}

export function CottonSfmFigure() {
  const [copy, zh] = useCopy();
  return (
    <InteractiveFigure
      title={copy.sfmTitle}
      caption={copy.sfmCaption(fmt(COTTON3D.sfm.count, zh))}
      load={loadViewer}
      loadProps={{ mode: "sfm" }}
      poster={useBaseUrl("/img/cotton3d/sfm.webp")}
      posterAlt={copy.posterAlt}
    />
  );
}

export function CottonCompareFigure() {
  const [copy, zh] = useCopy();
  return (
    <InteractiveFigure
      title={copy.compareTitle}
      caption={copy.compareCaption(
        fmt(COTTON3D.sfm.count, zh),
        fmt(COTTON3D.hy3d.count, zh),
        fmt(COTTON3D.hy3d.sourceCount, zh)
      )}
      load={loadViewer}
      loadProps={{ mode: "compare" }}
      poster={useBaseUrl("/img/cotton3d/compare.webp")}
      posterAlt={copy.posterAlt}
    />
  );
}

export function CottonCanopyFigure() {
  const [copy, zh] = useCopy();
  const c = COTTON3D.canopy;
  return (
    <InteractiveFigure
      title={copy.canopyTitle}
      caption={copy.canopyCaption(
        fmt(c.full.triangles, zh),
        fmt(c.leaf.triangles + c.nonleaf.triangles, zh),
        c.full.leafArea_m2,
        c.full.nonleafArea_m2
      )}
      load={loadViewer}
      loadProps={{ mode: "canopy" }}
      poster={useBaseUrl("/img/cotton3d/canopy.webp")}
      posterAlt={copy.posterAlt}
    />
  );
}
