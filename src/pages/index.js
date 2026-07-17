import React from "react";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Heading from "@theme/Heading";
import HomepageFeatures from "@site/src/components/HomepageFeatures";
import HologramParticles from "@site/src/components/HologramParticles";
import styles from "./index.module.css";

// pageTitle renders as "<pageTitle> | Liangchao Deng". It carries the topical
// keywords, since "Home" ranks for nothing.
const HOME_COPY = {
  en: {
    pageTitle: "AI for Plant Phenotyping & Crop Modeling",
    eyebrow: "Liangchao Deng · SMILER488",
    title: "Intelligent systems for crops, built to be explored.",
    role: "Postdoctoral Researcher · AI for Plant Phenotyping & Crop Modeling",
    intro:
      "I connect computer vision, crop models, and scientific AI to turn complex plant data into reproducible research tools.",
    chips: ["Plant Phenotyping", "Crop Modeling", "AI for Science"],
    primaryAction: "Explore the research",
    appAction: "Open App Lab",
    cvAction: "View CV",
    availability: "Research, open tools, and collaboration",
    description:
      "Postdoctoral research on AI for plant phenotyping, crop modeling, and computer vision — plus 14 free browser tools for field data, imaging, and analysis.",
  },
  zh: {
    pageTitle: "作物表型与作物模型的人工智能研究",
    eyebrow: "邓良超 · SMILER488",
    title: "把作物科学，构建成可以探索的智能系统。",
    role: "博士后研究人员 · 人工智能 × 作物表型 × 作物模型",
    intro:
      "连接计算机视觉、作物模型与科学智能，把复杂的植物数据转化为可复现、可使用的科研工具。",
    chips: ["作物表型", "作物模型", "科学智能"],
    primaryAction: "探索研究方向",
    appAction: "进入应用实验室",
    cvAction: "查看简历",
    availability: "研究、开放工具与合作",
    description:
      "面向作物表型、作物模型与计算机视觉的博士后研究，并提供 14 个免费浏览器工具，用于田间数据、图像分析与科研工作流。",
  },
};

function HomepageHeader() {
  const { i18n } = useDocusaurusContext();
  const isChinese = i18n.currentLocale === "zh-Hans";
  const copy = isChinese ? HOME_COPY.zh : HOME_COPY.en;
  const [showLoveEasterEgg, setShowLoveEasterEgg] = React.useState(false);

  const handleEasterEggTrigger = () => {
    setShowLoveEasterEgg(true);
    setTimeout(() => {
      window.location.href = "https://tangbonnie.github.io";
    }, 1600);
  };

  return (
    <header className={styles.heroBanner} data-particle-stage>
      <div className={styles.particleLayer}>
        <HologramParticles
          text="SMILER488"
          cloudImage="/img/cloud.png"
          cameraControls={false}
          obstacleSelector="[data-particle-obstacle]"
          style={{ width: "100%", height: "100%" }}
          onEasterEggTrigger={handleEasterEggTrigger}
        />
      </div>

      <div
        className={styles.heroContent}
        data-particle-obstacle
        aria-labelledby="home-hero-title"
      >
        <p className={styles.eyebrow}>{copy.eyebrow}</p>
        <Heading as="h1" id="home-hero-title" className={styles.heroTitle}>
          {copy.title}
        </Heading>
        <p className={styles.heroRole}>{copy.role}</p>
        <p className={styles.heroIntro}>{copy.intro}</p>

        <ul className={styles.chipList} aria-label={copy.role}>
          {copy.chips.map((chip) => (
            <li key={chip}>{chip}</li>
          ))}
        </ul>

        <div className={styles.actionRow}>
          <Link className={styles.primaryAction} to="#research-heading">
            {copy.primaryAction}
            <span aria-hidden="true">↘</span>
          </Link>
          <Link className={styles.secondaryAction} to="/app">
            {copy.appAction}
          </Link>
          <Link className={styles.textAction} to="/cv">
            {copy.cvAction}
            <span aria-hidden="true">→</span>
          </Link>
        </div>

        <p className={styles.availability}>
          <span aria-hidden="true" />
          {copy.availability}
        </p>
      </div>

      {showLoveEasterEgg && (
        <div className={styles.easterEggOverlay} data-particle-obstacle>
          <div className={styles.easterEggCard}>
            <div className={styles.easterEggHearts}>❤️ ✨ 💖 ✨ ❤️</div>
            <p className={styles.easterEggText}>
              {isChinese
                ? "正在穿梭前往 Bonnie 的空间..."
                : "Traversing to Bonnie's Space..."}
            </p>
          </div>
        </div>
      )}
    </header>
  );
}

export default function Home() {
  const { i18n } = useDocusaurusContext();
  const isChinese = i18n.currentLocale === "zh-Hans";
  const copy = isChinese ? HOME_COPY.zh : HOME_COPY.en;

  return (
    <Layout title={copy.pageTitle} description={copy.description}>
      <HomepageHeader />
      <main className={styles.homeMain}>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
