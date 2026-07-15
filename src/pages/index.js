import React from "react";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import HomepageFeatures from "@site/src/components/HomepageFeatures";
import Heading from "@theme/Heading";
import clsx from "clsx";
import styles from "./index.module.css";
import HologramParticles from "@site/src/components/HologramParticles";

const ZH_HOLOGRAM_LABELS = {
  pointerReady: "指针互动已就绪",
  reducedMotion: "静态模式 · 指针互动可用",
  cameraStarting: "正在启动本地手势识别…",
  cameraWaiting: "手势模式 · 请将一只手放入画面",
  cameraOpen: "张开手掌 · 粒子橡皮擦",
  cameraClosed: "闭合手势 · 粒子引力场",
  cameraError: "摄像头不可用 · 已保留指针模式",
  cameraUnsupported: "浏览器不支持摄像头 · 指针模式可用",
  enableCamera: "启用手势",
  disableCamera: "关闭摄像头",
  cancelCamera: "取消启动",
  retryCamera: "重试手势",
  unavailableCamera: "摄像头不可用",
  privacy: "仅在本机处理 · 不会上传视频",
  pointerHint: "移动或触摸擦除 · 鼠标按下聚合",
  eraserOpen: "擦除",
  eraserClosed: "引力",
};

// CloudAnimation 组件：在页面上显示移动云朵
function CloudAnimation() {
  const containerRef = React.useRef(null);
  const cloudRef = React.useRef(null);
  React.useEffect(() => {
    const container = containerRef.current;
    const cloud = cloudRef.current;
    if (!container || !cloud) return;

    let w = container.clientWidth;
    let h = container.clientHeight;
    let frameId = 0;
    let disposed = false;
    const motionPreference = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    );
    let reduceMotion = motionPreference.matches;
    const margin = 12;

    function bounds() {
      const cw = cloud.offsetWidth || 200;
      const ch = cloud.offsetHeight || 120;
      const s = getComputedStyle(container.parentElement);
      const leftVar = parseFloat(s.getPropertyValue("--grid-text-left"));
      const rightVar = parseFloat(s.getPropertyValue("--grid-text-right"));
      const hasVars =
        Number.isFinite(leftVar) &&
        Number.isFinite(rightVar) &&
        rightVar > leftVar;
      const xMin = hasVars
        ? Math.max(margin, leftVar)
        : Math.max(margin, w * 0.2);
      const xMaxRaw = hasVars ? rightVar : w * 0.8;
      const xMax = Math.max(
        xMin + 10,
        Math.min(w - cw - margin, xMaxRaw - cw - margin)
      );
      const yMin = Math.max(margin, h * 0.25);
      const yMax = Math.max(yMin + 10, h * 0.65 - ch - margin);
      return { xMin, xMax, yMin, yMax };
    }
    let b = bounds();

    let x = b.xMin + Math.random() * (b.xMax - b.xMin);
    let y = b.yMin + Math.random() * (b.yMax - b.yMin);
    let tx = x,
      ty = y;
    let start = 0;
    let dur = 6000;

    function pickTarget() {
      b = bounds();
      tx = b.xMin + Math.random() * (b.xMax - b.xMin);
      ty = b.yMin + Math.random() * (b.yMax - b.yMin);
      dur = 4000 + Math.random() * 6000;
      start = 0;
    }

    function step(t) {
      if (disposed || reduceMotion) return;
      if (!start) start = t;
      const p = Math.min(1, (t - start) / dur);
      const ease = 0.5 - Math.cos(Math.PI * p) / 2;
      const nx = x + (tx - x) * ease;
      const ny = y + (ty - y) * ease;
      cloud.style.transform = `translate(${nx}px, ${ny}px)`;
      if (p >= 1) {
        x = tx;
        y = ty;
        pickTarget();
      }
      frameId = requestAnimationFrame(step);
    }

    function positionStatic() {
      cancelAnimationFrame(frameId);
      frameId = 0;
      b = bounds();
      x = b.xMin + (b.xMax - b.xMin) * 0.72;
      y = b.yMin + (b.yMax - b.yMin) * 0.3;
      cloud.style.transform = `translate(${x}px, ${y}px)`;
    }

    function startAnimation() {
      cancelAnimationFrame(frameId);
      start = 0;
      pickTarget();
      frameId = requestAnimationFrame(step);
    }

    if (reduceMotion) positionStatic();
    else startAnimation();

    function onResize() {
      w = container.clientWidth;
      h = container.clientHeight;
      b = bounds();
      if (reduceMotion) positionStatic();
    }

    function onMotionPreferenceChange(event) {
      reduceMotion = event.matches;
      if (reduceMotion) positionStatic();
      else startAnimation();
    }

    window.addEventListener("resize", onResize);
    if (motionPreference.addEventListener) {
      motionPreference.addEventListener("change", onMotionPreferenceChange);
    } else {
      motionPreference.addListener?.(onMotionPreferenceChange);
    }

    return () => {
      disposed = true;
      cancelAnimationFrame(frameId);
      window.removeEventListener("resize", onResize);
      if (motionPreference.removeEventListener) {
        motionPreference.removeEventListener(
          "change",
          onMotionPreferenceChange
        );
      } else {
        motionPreference.removeListener?.(onMotionPreferenceChange);
      }
    };
  }, []);

  return (
    <div className={styles.cloudContainer} ref={containerRef}>
      <a
        ref={cloudRef}
        href="https://github.com/tangbonnie/tangbonnie.github.io"
        target="_blank"
        rel="noopener noreferrer"
        className={styles.cloudLink}
        title="Visit TangBonnie's GitHub (Click Me!)"
      >
        <img className={styles.cloud} src="/img/cloud.png" alt="Cloud" />
      </a>
    </div>
  );
}

function HomepageHeader() {
  const { siteConfig, i18n } = useDocusaurusContext();
  const isChinese = i18n.currentLocale === "zh-Hans";
  return (
    <header className={clsx("hero hero--primary", styles.heroBanner)}>
      <div className={styles.gridCanvas}>
        <HologramParticles
          text="SMILER488"
          style={{ height: "100%" }}
          labels={isChinese ? ZH_HOLOGRAM_LABELS : undefined}
        />
      </div>
      <CloudAnimation />
      <div className={styles.heroContent}>
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className={clsx("button button--secondary button--lg", styles.cta)}
            to="/cv"
            aria-label={isChinese ? "打开最新简历" : "Open the latest CV"}
            title={isChinese ? "打开最新简历" : "Open the latest CV"}
          >
            Curriculum Vitae - Latest
            <span className={styles.arrow}></span>
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="Deng Liangchao's research, computational plant science projects, and interactive tools."
    >
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
