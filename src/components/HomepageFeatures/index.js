import React from "react";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

const CONTENT = {
  en: {
    researchEyebrow: "Research directions",
    researchTitle: "From plant structure to predictive intelligence.",
    researchIntro:
      "Three connected layers turn field observations into measurable traits, explainable models, and useful decisions.",
    research: [
      {
        title: "Digital Crop Phenotyping",
        description:
          "Multi-view 3D reconstruction, UAV imaging, and computer vision for efficient, multi-scale measurement of crop structure and function.",
        imageSrc: "/img/compress_comic1.png",
        href: "/blog/tags/plant-phenotyping",
        meta: "Observe · Reconstruct · Quantify",
      },
      {
        title: "AI-powered Phenomic Analysis",
        description:
          "Computer vision and scientific AI that make phenotypic data processing more automatic, traceable, and reusable.",
        imageSrc: "/img/compress_comic2.png",
        href: "/blog/tags/artificial-intelligence",
        meta: "Vision · Language · Workflow",
      },
      {
        title: "Crop Modeling & Canopy Design",
        description:
          "Phenotypic and environmental data linked with crop and photosynthesis models for better canopy design and breeding decisions.",
        imageSrc: "/img/compress_comic3.png",
        href: "/blog/tags/crop-modeling",
        meta: "Model · Predict · Design",
      },
    ],
    labEyebrow: "Featured tools",
    labTitle: "Interactive computing, built to run in your browser.",
    labIntro:
      "Launch these crop analyses, field data capture, and scientific visualization tools directly.",
    lab: [
      {
        number: "01",
        title: "Sensor Recorder",
        description:
          "Capture device orientation, solar geometry, and GPS coordinates for leaf field measurements.",
        href: "/app/sensor",
        action: "Open tool",
        accent: "green",
      },
      {
        number: "02",
        title: "AI Data Visualizer",
        description:
          "Upload and analyze crop datasets interactively with publication-ready scientific plotting.",
        href: "/app/ai-data-visualizer",
        action: "Open tool",
        accent: "blue",
      },
      {
        number: "03",
        title: "Root Preprocessor",
        description:
          "Clean, segment, and preprocess root system imagery to extract morphology phenotypes.",
        href: "/app/root-processor",
        action: "Open tool",
        accent: "orange",
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
    researchEyebrow: "研究方向",
    researchTitle: "从植物结构，走向可预测的智能。",
    researchIntro:
      "把田间观测连接为可测量的表型、可解释的模型与可执行的科研决策。",
    research: [
      {
        title: "数字作物表型",
        description:
          "融合多视角三维重建、无人机成像与计算机视觉，高效量化作物多尺度结构与功能。",
        imageSrc: "/img/compress_comic1.png",
        href: "/blog/tags/plant-phenotyping",
        meta: "观测 · 重建 · 量化",
      },
      {
        title: "AI 驱动的表型组分析",
        description:
          "以计算机视觉和科学智能提升表型数据处理的自动化、可追溯性与复用能力。",
        imageSrc: "/img/compress_comic2.png",
        href: "/blog/tags/artificial-intelligence",
        meta: "视觉 · 语言 · 工作流",
      },
      {
        title: "作物模型与冠层设计",
        description:
          "连接表型、环境数据与作物及光合模型，为高效冠层设计和育种决策提供依据。",
        imageSrc: "/img/compress_comic3.png",
        href: "/blog/tags/crop-modeling",
        meta: "建模 · 预测 · 设计",
      },
    ],
    labEyebrow: "精选工具",
    labTitle: "交互式计算，在浏览器中即点即用。",
    labIntro: "直接启动以下作物分析、数据规划与科学可视化应用。",
    lab: [
      {
        number: "01",
        title: "Sensor Recorder",
        description:
          "获取手机姿态、太阳入射角与定位信息，辅助田间叶片测量与数据采集。",
        href: "/app/sensor",
        action: "打开工具",
        accent: "green",
      },
      {
        number: "02",
        title: "AI Data Visualizer",
        description:
          "交互式上传分析作物科学数据集，快速生成可供发表的学术图表。",
        href: "/app/ai-data-visualizer",
        action: "打开工具",
        accent: "blue",
      },
      {
        number: "03",
        title: "Root Preprocessor",
        description:
          "清理、分割与处理作物根系图像，快速提取和量化根系形态学特征。",
        href: "/app/root-processor",
        action: "打开工具",
        accent: "orange",
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

function ResearchCard({ item }) {
  const imageNumber = item.imageSrc.match(/comic(\d)/)?.[1];
  const responsiveSource = imageNumber
    ? `/img/home-comic${imageNumber}-768.webp 768w, /img/home-comic${imageNumber}-1200.webp 1200w`
    : undefined;

  return (
    <Link className={styles.researchCard} to={item.href}>
      <div className={styles.researchVisual}>
        <picture>
          {responsiveSource && (
            <source
              type="image/webp"
              srcSet={responsiveSource}
              sizes="(max-width: 700px) calc(100vw - 2rem), (max-width: 996px) 42vw, 30vw"
            />
          )}
          <img
            src={item.imageSrc}
            alt=""
            width="2048"
            height="2048"
            loading="lazy"
            decoding="async"
          />
        </picture>
        <span className={styles.researchMeta}>{item.meta}</span>
      </div>
      <div className={styles.researchBody}>
        <Heading as="h3">{item.title}</Heading>
        <p>{item.description}</p>
        <span className={styles.cardArrow} aria-hidden="true">
          ↗
        </span>
      </div>
    </Link>
  );
}

function LabCard({ item }) {
  return (
    <Link
      className={styles.labCard}
      data-accent={item.accent}
      to={item.href}
    >
      <div className={styles.labCardTop}>
        <span className={styles.labNumber}>{item.number}</span>
        <span className={styles.labGlyph} aria-hidden="true" />
      </div>
      <div>
        <Heading as="h3">{item.title}</Heading>
        <p>{item.description}</p>
      </div>
      <span className={styles.labAction}>
        {item.action}
        <span aria-hidden="true">→</span>
      </span>
    </Link>
  );
}

export default function HomepageFeatures() {
  const { i18n } = useDocusaurusContext();
  const isChinese = i18n.currentLocale === "zh-Hans";
  const copy = isChinese ? CONTENT.zh : CONTENT.en;

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
          <div className={styles.researchGrid}>
            {copy.research.map((item) => (
              <ResearchCard key={item.title} item={item} />
            ))}
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
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: '0.2rem' }}>
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                  <polyline points="14 2 14 8 20 8"></polyline>
                  <line x1="16" y1="13" x2="8" y2="13"></line>
                  <line x1="16" y1="17" x2="8" y2="17"></line>
                  <polyline points="10 9 9 9 8 9"></polyline>
                </svg>
                {isChinese ? "研究笔记" : "Research Notes"}
              </Link>
              <Link className={styles.portalLink} to="/resources">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: '0.2rem' }}>
                  <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                  <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
                </svg>
                {isChinese ? "学习资源" : "Resources"}
              </Link>
              <Link className={styles.portalLink} to="/cv">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: '0.2rem' }}>
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
              <span aria-hidden="true">↗</span>
            </Link>
            <Link
              className={styles.commercialAction}
              to="mailto:dengliangchao@smiler488.com"
            >
              {copy.commercialAction}
              <span aria-hidden="true">↗</span>
            </Link>
            <Link
              className={styles.botAction}
              to="https://work.weixin.qq.com/kfid/kfc63941027aeefc636"
              target="_blank"
              rel="noopener noreferrer"
            >
              {copy.botAction}
              <span aria-hidden="true">↗</span>
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}
