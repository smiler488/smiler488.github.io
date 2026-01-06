import React from "react";
import Layout from "@theme/Layout";
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import clsx from 'clsx';
import styles from './index.module.css';
import HologramParticles from '@site/src/components/HologramParticles';

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
    // Note: cloud is now the <a> tag, checking its dimensions might need its child img loaded or preset size
    // We can assume a default size or base it on the image styles
    const cw = 200; // approximated width from CSS
    const ch = 120; // approximated height
    const margin = 12;

    function bounds() {
      const s = getComputedStyle(container.parentElement);
      const leftVar = parseFloat(s.getPropertyValue('--grid-text-left'));
      const rightVar = parseFloat(s.getPropertyValue('--grid-text-right'));
      const hasVars = Number.isFinite(leftVar) && Number.isFinite(rightVar) && rightVar > leftVar;
      const xMin = hasVars ? Math.max(margin, leftVar) : Math.max(margin, w * 0.2);
      const xMaxRaw = hasVars ? rightVar : w * 0.8;
      const xMax = Math.max(xMin + 10, Math.min(w - cw - margin, xMaxRaw - cw - margin));
      const yMin = Math.max(margin, h * 0.25);
      const yMax = Math.max(yMin + 10, h * 0.65 - ch - margin);
      return { xMin, xMax, yMin, yMax };
    }
    let b = bounds();

    let x = b.xMin + Math.random() * (b.xMax - b.xMin);
    let y = b.yMin + Math.random() * (b.yMax - b.yMin);
    let tx = x, ty = y;
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
      if (!start) start = t;
      const p = Math.min(1, (t - start) / dur);
      const ease = 0.5 - Math.cos(Math.PI * p) / 2;
      const nx = x + (tx - x) * ease;
      const ny = y + (ty - y) * ease;
      cloud.style.transform = `translate(${nx}px, ${ny}px)`;
      if (p >= 1) {
        x = tx; y = ty;
        pickTarget();
      }
      requestAnimationFrame(step);
    }
    pickTarget();
    requestAnimationFrame(step);

    function onResize() {
      w = container.clientWidth;
      h = container.clientHeight;
      b = bounds();
    }
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
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
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className={styles.gridCanvas}>
        <HologramParticles text="SMILER488" style={{ height: '100%', pointerEvents: 'none' }} />
      </div>
      <CloudAnimation />
      <div className={styles.heroContent}>
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className={clsx('button button--secondary button--lg', styles.cta)}
            to="/cv"
            aria-label="打开最新简历"
            title="打开最新简历">
            Curriculum Vitae - Latest
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
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
