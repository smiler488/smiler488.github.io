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
    labEyebrow: "Explore the lab",
    labTitle: "Research is more useful when you can interact with it.",
    labIntro:
      "Open the tools, inspect the methods, follow a learning path, or review the complete academic profile.",
    lab: [
      {
        number: "01",
        title: "App Lab",
        description:
          "Fourteen browser-based tools for imaging, phenotyping, scientific writing, data analysis, and local AI workflows.",
        href: "/app",
        action: "Launch apps",
        accent: "blue",
      },
      {
        number: "02",
        title: "Research Notes",
        description:
          "Field-tested notes on computer vision, crop research, reproducible computing, and practical AI engineering.",
        href: "/blog",
        action: "Read notes",
        accent: "violet",
      },
      {
        number: "03",
        title: "Learning Resources",
        description:
          "Curated paths for AI for Science, crop modeling, plant phenotyping, and research software.",
        href: "/resources",
        action: "Follow a path",
        accent: "green",
      },
      {
        number: "04",
        title: "Curriculum Vitae",
        description:
          "Education, research experience, publications, projects, skills, and current contact channels.",
        href: "/cv",
        action: "Open CV",
        accent: "orange",
      },
    ],
    collaborationEyebrow: "Work together",
    collaborationTitle: "Have a crop-science problem worth making computable?",
    collaborationText:
      "I welcome research exchange, open-source collaboration, and carefully scoped commercial projects.",
    academicAction: "Academic contact",
    commercialAction: "Commercial cooperation",
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
    labEyebrow: "探索实验室",
    labTitle: "科研成果，在可以交互时更有价值。",
    labIntro: "使用工具、查看方法、跟随学习路径，或了解完整的学术经历。",
    lab: [
      {
        number: "01",
        title: "应用实验室",
        description:
          "十四个浏览器端工具，覆盖图像、表型、科研写作、数据分析与本地 AI 工作流。",
        href: "/app",
        action: "启动应用",
        accent: "blue",
      },
      {
        number: "02",
        title: "研究笔记",
        description:
          "记录计算机视觉、作物研究、可复现计算与实用 AI 工程中的方法和经验。",
        href: "/blog",
        action: "阅读笔记",
        accent: "violet",
      },
      {
        number: "03",
        title: "学习资源",
        description:
          "面向 AI for Science、作物模型、植物表型与科研软件的精选学习路径。",
        href: "/resources",
        action: "开始学习",
        accent: "green",
      },
      {
        number: "04",
        title: "个人简历",
        description: "教育、科研经历、论文、项目、技能与当前有效的联系渠道。",
        href: "/cv",
        action: "查看简历",
        accent: "orange",
      },
    ],
    collaborationEyebrow: "一起工作",
    collaborationTitle: "有一个值得被计算化的作物科学问题？",
    collaborationText:
      "欢迎科研交流、开源协作，以及范围清晰、目标明确的商业合作。",
    academicAction: "学术联系",
    commercialAction: "商业合作",
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

function LabCard({ item, index }) {
  return (
    <Link
      className={styles.labCard}
      data-accent={item.accent}
      data-size={index < 2 ? "wide" : "compact"}
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
            {copy.lab.map((item, index) => (
              <LabCard key={item.title} item={item} index={index} />
            ))}
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
          </div>
        </div>
      </section>
    </>
  );
}
