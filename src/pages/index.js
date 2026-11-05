import React from "react";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Heading from "@theme/Heading";
import useBaseUrl from "@docusaurus/useBaseUrl";
import HomepageFeatures from "@site/src/components/HomepageFeatures";
import styles from "./index.module.css";
import { PersonJsonLd } from "@site/src/components/ds";

// pageTitle renders as "<pageTitle> | Liangchao Deng". It carries the topical
// keywords, since "Home" ranks for nothing.
const HOME_COPY = {
  en: {
    pageTitle: "AI4Plant · Plant Phenotyping & Crop Modeling",
    particleText: "Liangchao Deng",
    eyebrow: "Liangchao Deng",
    emblemAlt:
      "AI4Plant emblem: a researcher with a tablet in a crop field, framed by satellite and drone remote sensing, 3D plant reconstruction, leaf BRDF and root growth panels.",
    intro:
      "I connect computer vision, crop models, and scientific AI to turn complex plant data into reproducible research tools.",
    primaryAction: "Explore the research",
    appAction: "Open App Lab",
    cvAction: "View CV",
    description:
      "AI4Plant research on plant phenotyping, crop modeling, and computer vision — plus 15 free browser tools for field data, imaging, and analysis.",
  },
  zh: {
    pageTitle: "AI4Plant · 作物表型与作物模型",
    particleText: "邓良超",
    eyebrow: "邓良超",
    emblemAlt:
      "AI4Plant 标志：田间手持平板的研究者，周围是卫星与无人机遥感、三维植株重建、叶片 BRDF 与根系生长等画面。",
    intro:
      "连接计算机视觉、作物模型与科学智能，把复杂的植物数据转化为可复现、可使用的科研工具。",
    primaryAction: "探索研究方向",
    appAction: "进入应用实验室",
    cvAction: "查看简历",
    description:
      "AI4Plant：面向作物表型、作物模型与计算机视觉的研究，并提供 15 个免费浏览器工具，用于田间数据、图像分析与科研工作流。",
  },
};

function HomepageHeader() {
  const { i18n } = useDocusaurusContext();
  const isChinese = i18n.currentLocale === "zh-Hans";
  const copy = isChinese ? HOME_COPY.zh : HOME_COPY.en;
  const [showLoveEasterEgg, setShowLoveEasterEgg] = React.useState(false);
  const emblem = useBaseUrl("/img/ai4plant-emblem-360.webp");
  const emblem2x = useBaseUrl("/img/ai4plant-emblem-720.webp");

  const handleEasterEggTrigger = () => {
    setShowLoveEasterEgg(true);
    setTimeout(() => {
      window.location.href = "https://tangbonnie.github.io";
    }, 1600);
  };

  // Preserve the hidden double-click easter egg from the old particle hero:
  // double-clicking empty hero space (not the content or a link) traverses
  // to Bonnie's space.
  const handleHeroDoubleClick = (event) => {
    const target = event.target;
    if (
      target instanceof Element &&
      target.closest(
        "a,button,input,textarea,select,summary,[role='button'],[data-particle-obstacle]"
      )
    ) {
      return;
    }
    handleEasterEggTrigger();
  };

  return (
    <header
      className={styles.heroBanner}
      onDoubleClick={handleHeroDoubleClick}
    >
      <div
        className={styles.heroContent}
        data-particle-obstacle
        aria-labelledby="home-hero-title"
      >
        {/* Copy column: eyebrow, the name as the page heading, intro, one
            blue action and two circled-arrow links (muse-design page layer) */}
        <div className={styles.heroCopy}>
          <p className={styles.heroEyebrow}>
            AI<span className={styles.four}>4</span>Plant
          </p>
          <Heading as="h1" id="home-hero-title" className={styles.heroName}>
            {copy.eyebrow}
          </Heading>
          <p className={styles.heroIntro}>{copy.intro}</p>

          <div className={styles.actionRow}>
            <Link className={styles.primaryAction} to="#research-heading">
              {copy.primaryAction}
            </Link>
            <Link className={styles.arrowAction} to="/app">
              {copy.appAction}
            </Link>
            <Link className={styles.arrowAction} to="/cv">
              {copy.cvAction}
            </Link>
          </div>
        </div>

        {/* Media column: the emblem in a soft panel, with its name capsule */}
        <div className={styles.heroMedia}>
          <img
            className={styles.emblem}
            src={emblem}
            srcSet={`${emblem} 360w, ${emblem2x} 720w`}
            sizes="(max-width: 860px) 60vw, 360px"
            width="360"
            height="360"
            alt={copy.emblemAlt}
          />
          <span className={styles.identityTag}>
            <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
              <path d="M5 19C5 10.5 11 5 20 5c0 8.5-6 14-15 14Z" />
              <path d="M5 19 13 11" />
            </svg>
            <span>
              AI<span className={styles.four}>4</span>Plant
            </span>
          </span>
        </div>
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
      <PersonJsonLd />
      <HomepageHeader />
      <main className={styles.homeMain}>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
