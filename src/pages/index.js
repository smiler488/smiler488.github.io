import React from "react";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Heading from "@theme/Heading";
import HomepageFeatures from "@site/src/components/HomepageFeatures";
import HologramParticles from "@site/src/components/HologramParticles";
import styles from "./index.module.css";

const HOME_COPY = {
  en: {
    pageTitle: "Home",
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
      "Liangchao Deng's research, computational plant science projects, and interactive tools.",
  },
  zh: {
    pageTitle: "首页",
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
    description: "邓良超的作物表型、作物模型、科学智能研究与交互工具。",
  },
};

const ZH_HOLOGRAM_LABELS = {
  pointerReady: "粒子互动已就绪",
  reducedMotion: "静态粒子展示 · 已减少动态效果",
  cameraStarting: "正在启动本地手势识别…",
  cameraPreviewOnly: "摄像头背景已开启 · 指针互动仍可用",
  cameraWaiting: "手势模式 · 请将一只手放入画面",
  cameraOpen: "张开手掌 · 扩散粒子",
  cameraClosed: "闭合手势 · 吸附并旋转粒子",
  cameraError: "摄像头不可用 · 已保留指针模式",
  cameraUnsupported: "浏览器不支持摄像头 · 指针模式可用",
  enableCamera: "启用手势",
  disableCamera: "关闭摄像头",
  cancelCamera: "取消启动",
  retryCamera: "重试手势",
  unavailableCamera: "摄像头不可用",
  privacy: "仅在本机处理 · 不会上传视频",
  pointerHint: "移动或触摸扩散 · 按下或切换模式吸附",
  fieldDisperse: "切换为粒子扩散模式",
  fieldAttract: "切换为粒子吸附模式",
  fieldModeDisperse: "扩散",
  fieldModeAttract: "吸附",
};

function HomepageHeader() {
  const { i18n } = useDocusaurusContext();
  const isChinese = i18n.currentLocale === "zh-Hans";
  const copy = isChinese ? HOME_COPY.zh : HOME_COPY.en;

  return (
    <header className={styles.heroBanner} data-particle-stage>
      <div className={styles.particleLayer}>
        <HologramParticles
          text="SMILER488"
          cloudImage="/img/cloud.png"
          showCameraPreview
          obstacleSelector="[data-particle-obstacle]"
          style={{ height: "100%" }}
          labels={isChinese ? ZH_HOLOGRAM_LABELS : undefined}
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
