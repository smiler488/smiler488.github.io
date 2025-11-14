import React from "react";
import Layout from "@theme/Layout";
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import clsx from 'clsx';
import styles from './index.module.css';

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
    const cw = cloud.clientWidth || 200;
    const ch = cloud.clientHeight || 120;
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
      <a href="https://github.com/tangbonnie/tangbonnie.github.io" target="_blank" rel="noopener noreferrer" style={{ pointerEvents: 'auto' }}>
        <img ref={cloudRef} className={styles.cloud} src="/img/cloud.png" alt="Cloud" />
      </a>
    </div>
  );
}

function GridBackground() {
  const canvasRef = React.useRef(null);
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    function draw() {
      const w = parent.clientWidth;
      const h = parent.clientHeight;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = w + 'px';
      canvas.style.height = h + 'px';
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const styles = getComputedStyle(document.documentElement);
      const bg = styles.getPropertyValue('--home-grid-bg').trim() || '#e6e7ea';
      const line = styles.getPropertyValue('--home-grid-line').trim() || '#c8c9cc';
      const ink = styles.getPropertyValue('--home-grid-letter').trim() || '#000000';

      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, w, h);

      // text metrics
      const text = 'smiler488';
      const glyphW = 5, glyphH = 7, spacing = 2; // in cells
      const shear = -0.25; // italic shear factor
      const totalCellsW = text.length * glyphW + (text.length - 1) * spacing;

      // choose cell size relative to viewport so text always fits
      const widthCellMax = (w * 0.9) / (totalCellsW + Math.abs(shear) * glyphH);
      const heightCellMax = (h * 0.45) / glyphH;
      const cell = Math.max(8, Math.floor(Math.min(widthCellMax, heightCellMax))); // responsive cell

      // draw grid
      ctx.strokeStyle = line;
      ctx.lineWidth = 1;
      for (let x = 0; x <= w; x += cell) {
        ctx.beginPath(); ctx.moveTo(x + 0.5, 0); ctx.lineTo(x + 0.5, h); ctx.stroke();
      }
      for (let y = 0; y <= h; y += cell) {
        ctx.beginPath(); ctx.moveTo(0, y + 0.5); ctx.lineTo(w, y + 0.5); ctx.stroke();
      }

      // 5x7 pixel font for characters
      const font = {
        's': [
          '11110',
          '1....',
          '11110',
          '....1',
          '11110',
          '.....',
          '.....'],
        'm': [
          '1.1.1',
          '11111',
          '1...1',
          '1...1',
          '1...1',
          '.....',
          '.....'],
        'i': [
          '..1..',
          '..1..',
          '..1..',
          '..1..',
          '..1..',
          '.....',
          '.....'],
        'l': [
          '1....',
          '1....',
          '1....',
          '1....',
          '11110',
          '.....',
          '.....'],
        'e': [
          '11110',
          '1....',
          '11110',
          '1....',
          '11110',
          '.....',
          '.....'],
        'r': [
          '11110',
          '1...1',
          '11110',
          '1.1..',
          '1..1.',
          '.....',
          '.....'],
        '4': [
          '...1.',
          '..11.',
          '.1.1.',
          '11111',
          '...1.',
          '.....',
          '.....'],
        '8': [
          '11111',
          '1...1',
          '11111',
          '1...1',
          '11111',
          '.....',
          '.....'],
      };
      // position text (center horizontally considering shear, upper region vertically)
      const naturalWidth = totalCellsW * cell;
      const baseY = Math.floor((h * 0.5 - glyphH * cell) / 2);
      const yMid = baseY + (glyphH * cell) / 2;
      const baseX = Math.floor(w / 2 - naturalWidth / 2 - shear * yMid);

      ctx.fillStyle = ink;
      ctx.save();
      // italic shear for letters
      ctx.transform(1, 0, shear, 1, 0, 0);
      for (let idx = 0, cursorCells = 0; idx < text.length; idx++) {
        const ch = text[idx];
        const mat = font[ch];
        if (mat) {
          for (let ry = 0; ry < glyphH; ry++) {
            const row = mat[ry];
            for (let rx = 0; rx < glyphW; rx++) {
              if (row && row[rx] === '1') {
                const x = baseX + (cursorCells + rx) * cell + 1;
                const y = baseY + ry * cell + 1;
                // bold by slightly larger fill
                ctx.fillRect(x, y, cell - 1, cell - 1);
                ctx.fillRect(x + 1, y, cell - 2, cell - 1);
              }
            }
          }
        }
        cursorCells += glyphW + spacing;
      }
      ctx.restore();

      // push hero content below the grid text dynamically
      const heroOffset = Math.min(h - 120, baseY + glyphH * cell + Math.round(0.8 * cell));
      document.documentElement.style.setProperty('--hero-content-offset', heroOffset + 'px');

      // expose horizontal bounds of the sheared text for other elements (e.g., cloud animation)
      const textLeft = Math.floor(baseX + shear * (baseY + glyphH * cell));
      const textRight = Math.floor(baseX + naturalWidth + shear * baseY);
      parent.style.setProperty('--grid-text-left', textLeft + 'px');
      parent.style.setProperty('--grid-text-right', textRight + 'px');
    }
    draw();
    const onResize = () => draw();
    const onTheme = () => draw();
    window.addEventListener('resize', onResize);
    const obs = new MutationObserver(onTheme);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => { window.removeEventListener('resize', onResize); obs.disconnect(); };
  }, []);
  return <canvas className={styles.gridCanvas} ref={canvasRef} />;
}

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <GridBackground />
      {/* 在头部添加云朵动画 */}
      <CloudAnimation />
      <div className={styles.heroContent}>
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className={clsx('button button--secondary button--lg', styles.cta)}
            to="/blog/curriculum-vitae"
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
  const {siteConfig} = useDocusaurusContext();
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
