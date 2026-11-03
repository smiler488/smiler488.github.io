import React from "react";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Heading from "@theme/Heading";
import LayerMorph from "@site/src/components/LayerMorph";

// Each animation step links to the research hub filtered to that layer.
const LAYER_EVIDENCE = ["DIG", "UND", "PRE", "DES"].map(
  (id) => `/research?layer=${id}`
);
import styles from "./styles.module.css";

const CONTENT = {
  en: {
    researchEyebrow: "Research architecture",
    researchTitle: "From sensing crops to designing crops.",
    researchIntro:
      "A four-layer architecture for crop intelligence — digitize the physical crop, understand its mechanisms, predict its future, and design what it should become.",
    steps: ["Digitize", "Understand", "Predict", "Design"],
    stepEvidence: (step) => `Evidence for ${step}`,
    research: [
      {
        mark: "I",
        tag: "Digitize",
        comic: "/img/research/layer-1-digitize.webp",
        alt: "Digitize the physical crop: multi-sensor field imaging (RGB, multispectral, thermal, LiDAR) and UAV scanning turned into 3D reconstructions and organ-level traits",
        href: "/research?layer=DIG",
      },
      {
        mark: "II",
        tag: "Understand",
        comic: "/img/research/layer-2-understand.webp",
        alt: "Understand its mechanisms: how leaf optics, canopy light distribution, 3D plant structure, gas exchange and physiology determine photosynthesis and growth",
        href: "/research?layer=UND",
      },
      {
        mark: "III",
        tag: "Predict",
        comic: "/img/research/layer-3-predict.webp",
        alt: "Predict its future: crop growth models and AI combining weather, soil, management, genotype and remote sensing to forecast photosynthesis and yield under future climates",
        href: "/research?layer=PRE",
      },
      {
        mark: "IV",
        tag: "Design",
        comic: "/img/research/layer-4-design.webp",
        alt: "Design what it should become: ideotype design, in silico evaluation and multi-objective optimization for high yield, resilience and resource efficiency",
        href: "/research?layer=DES",
      },
    ],
    labEyebrow: "Featured tools",
    labTitle: "Interactive computing, built to run in your browser.",
    labIntro:
      "Launch these crop analyses, field data capture, and scientific visualization tools directly.",
    lab: [
      {
        title: "Sensor Recorder",
        shot: "/img/app-shots/sensor.jpg",
        description:
          "Capture device orientation, solar geometry, and GPS coordinates for leaf field measurements.",
        href: "/app/sensor",
        action: "Open tool",
      },
      {
        title: "AI Data Visualizer",
        shot: "/img/app-shots/ai-data-visualizer.jpg",
        description:
          "Upload and analyze crop datasets interactively with publication-ready scientific plotting.",
        href: "/app/ai-data-visualizer",
        action: "Open tool",
      },
      {
        title: "Root Preprocessor",
        shot: "/img/app-shots/root-processor.jpg",
        description:
          "Clean, segment, and preprocess root system imagery to extract morphology phenotypes.",
        href: "/app/root-processor",
        action: "Open tool",
      },
    ],
    collaborationEyebrow: "Work together",
    collaborationTitle: "Have a crop-science problem worth making computable?",
    collaborationText:
      "I welcome research exchange, open-source collaboration, and carefully scoped commercial projects.",
    academicAction: "Academic contact",
    commercialAction: "Commercial cooperation",
    botAction: "WeChat Consulting Bot",
  },
  zh: {
    researchEyebrow: "研究架构",
    researchTitle: "从感知作物，走向设计作物。",
    researchIntro:
      "作物智能的四层架构——把物理作物数字化、理解其机理、预测其未来，并设计它应有的样子。",
    steps: ["数字化", "理解", "预测", "设计"],
    stepEvidence: (step) => `查看「${step}」层的证据`,
    research: [
      {
        mark: "I",
        tag: "数字化",
        comic: "/img/research/layer-1-digitize.webp",
        alt: "把物理作物数字化：田间多传感器成像（RGB、多光谱、热红外、激光雷达）与无人机扫描，转化为三维重建和器官级性状",
        href: "/research?layer=DIG",
      },
      {
        mark: "II",
        tag: "理解",
        comic: "/img/research/layer-2-understand.webp",
        alt: "理解其机理：叶片光学特性、冠层光分布、三维株型、气体交换与生理过程如何共同决定光合与生长",
        href: "/research?layer=UND",
      },
      {
        mark: "III",
        tag: "预测",
        comic: "/img/research/layer-3-predict.webp",
        alt: "预测其未来：作物生长模型与人工智能融合气象、土壤、管理、基因型和遥感数据，预测未来气候下的光合与产量",
        href: "/research?layer=PRE",
      },
      {
        mark: "IV",
        tag: "设计",
        comic: "/img/research/layer-4-design.webp",
        alt: "设计它应有的样子：理想株型设计、计算机模拟评估与多目标优化，面向高产、抗逆与资源高效",
        href: "/research?layer=DES",
      },
    ],
    labEyebrow: "精选工具",
    labTitle: "交互式计算，在浏览器中即点即用。",
    labIntro: "直接启动以下作物分析、数据规划与科学可视化应用。",
    lab: [
      {
        title: "Sensor Recorder",
        shot: "/img/app-shots/sensor.jpg",
        description:
          "获取手机姿态、太阳入射角与定位信息，辅助田间叶片测量与数据采集。",
        href: "/app/sensor",
        action: "打开工具",
      },
      {
        title: "AI Data Visualizer",
        shot: "/img/app-shots/ai-data-visualizer.jpg",
        description:
          "交互式上传分析作物科学数据集，快速生成可供发表的学术图表。",
        href: "/app/ai-data-visualizer",
        action: "打开工具",
      },
      {
        title: "Root Preprocessor",
        shot: "/img/app-shots/root-processor.jpg",
        description:
          "清理、分割与处理作物根系图像，快速提取和量化根系形态学特征。",
        href: "/app/root-processor",
        action: "打开工具",
      },
    ],
    collaborationEyebrow: "一起工作",
    collaborationTitle: "有一个值得被计算化的作物科学问题？",
    collaborationText:
      "欢迎科研交流、开源协作，以及范围清晰、目标明确的商业合作。",
    academicAction: "学术联系",
    commercialAction: "商业合作",
    botAction: "微信咨询机器人",
  },
};

// The four layer drawings stacked in one frame; only the current layer is
// visible and reachable. The others stay mounted so switching is instant.
function LayerShowcase({ items, active }) {
  return (
    <div className={styles.showcase}>
      {items.map((item, index) => (
        <ComicSlide key={item.mark} item={item} current={index === active} />
      ))}
    </div>
  );
}

function ComicSlide({ item, current }) {
  return (
    <Link
      className={`${styles.comicCard} ${current ? styles.comicCurrent : ""}`}
      to={item.href}
      aria-hidden={current ? undefined : true}
      tabIndex={current ? undefined : -1}
    >
      <img
        className={styles.comicImg}
        src={item.comic}
        alt={item.alt}
        width="1254"
        height="1254"
        loading="lazy"
        decoding="async"
      />
      <span className={styles.comicTag}>
        <span className={styles.comicTagMark} aria-hidden="true">
          {item.mark}
        </span>
        {item.tag}
      </span>
    </Link>
  );
}

function LabCard({ item }) {
  return (
    <Link className={styles.labCard} to={item.href}>
      <img
        className={styles.labShot}
        src={item.shot}
        alt=""
        width="800"
        height="500"
        loading="lazy"
        decoding="async"
      />
      <div className={styles.labCardBody}>
        <Heading as="h3">{item.title}</Heading>
        <p>{item.description}</p>
      </div>
      <span className={styles.labAction}>{item.action}</span>
    </Link>
  );
}

export default function HomepageFeatures() {
  const { i18n } = useDocusaurusContext();
  const isChinese = i18n.currentLocale === "zh-Hans";
  const copy = isChinese ? CONTENT.zh : CONTENT.en;
  const [layer, setLayer] = React.useState(0);

  return (
    <>
      <section
        id="research"
        className={styles.researchSection}
        aria-labelledby="research-heading"
      >
        <div className={styles.sectionShell}>
          <div className={styles.sectionHeader}>
            <p className={styles.eyebrow}>{copy.researchEyebrow}</p>
            <Heading as="h2" id="research-heading">
              {copy.researchTitle}
            </Heading>
            <p>{copy.researchIntro}</p>
          </div>
          <div className={styles.layerStage}>
            <LayerMorph
              steps={copy.steps}
              label={copy.researchTitle}
              evidenceHrefs={LAYER_EVIDENCE}
              evidenceLabel={copy.stepEvidence}
              onActiveChange={setLayer}
            />
            <LayerShowcase items={copy.research} active={layer} />
          </div>
        </div>
      </section>

      <section className={styles.labSection} aria-labelledby="lab-heading">
        <div className={styles.sectionShell}>
          <div className={styles.sectionHeader}>
            <p className={styles.eyebrow}>{copy.labEyebrow}</p>
            <Heading as="h2" id="lab-heading">
              {copy.labTitle}
            </Heading>
            <p>{copy.labIntro}</p>
          </div>
          <div className={styles.labGrid}>
            {copy.lab.map((item) => (
              <LabCard key={item.title} item={item} />
            ))}
          </div>

          <div className={styles.portalDockContainer}>
            <div className={styles.portalDock}>
              <span className={styles.portalLabel}>
                {isChinese ? "快捷入口" : "Portals"}
              </span>
              <Link className={styles.portalLink} to="/blog">
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  style={{ marginRight: "0.2rem" }}
                >
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                  <polyline points="14 2 14 8 20 8"></polyline>
                  <line x1="16" y1="13" x2="8" y2="13"></line>
                  <line x1="16" y1="17" x2="8" y2="17"></line>
                  <polyline points="10 9 9 9 8 9"></polyline>
                </svg>
                {isChinese ? "研究笔记" : "Research Notes"}
              </Link>
              <Link className={styles.portalLink} to="/resources">
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  style={{ marginRight: "0.2rem" }}
                >
                  <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                  <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
                </svg>
                {isChinese ? "学习资源" : "Resources"}
              </Link>
              <Link className={styles.portalLink} to="/cv">
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  style={{ marginRight: "0.2rem" }}
                >
                  <rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect>
                  <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path>
                </svg>
                {isChinese ? "个人简历" : "Curriculum Vitae"}
              </Link>
            </div>
          </div>
        </div>
      </section>

      <section
        className={styles.collaborationSection}
        aria-labelledby="collaboration-heading"
      >
        <div className={styles.collaborationCard}>
          <div>
            <p className={styles.eyebrow}>{copy.collaborationEyebrow}</p>
            <Heading as="h2" id="collaboration-heading">
              {copy.collaborationTitle}
            </Heading>
            <p>{copy.collaborationText}</p>
          </div>
          <div className={styles.contactActions}>
            <Link
              className={styles.academicAction}
              to="mailto:googalphdlc@gmail.com"
            >
              {copy.academicAction}
            </Link>
            <Link
              className={styles.commercialAction}
              to="mailto:dengliangchao@azureaxion.com"
            >
              {copy.commercialAction}
            </Link>
            <Link
              className={styles.botAction}
              to="https://work.weixin.qq.com/kfid/kfc63941027aeefc636"
              target="_blank"
              rel="noopener noreferrer"
            >
              {copy.botAction}
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}
