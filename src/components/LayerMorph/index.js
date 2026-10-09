import React from "react";
import Link from "@docusaurus/Link";
import styles from "./styles.module.css";

/**
 * LayerMorph — the four-layer architecture told as one continuous morph.
 *
 * ~150 points glide between four forms (Digitize → Understand → Predict →
 * Design) on cubic-bezier(.45,0,.15,1), staggered left to right. Each form's
 * overlay (scan line, network, growth curve, symmetry axis) dissolves in on
 * the back half of the glide into it and out on the front half of the glide
 * away from it. The canvas is decorative; the stepper below is the real
 * content and also drives the morph when clicked.
 */

const NUMERALS = ["I", "II", "III", "IV"];
const N = 150;
const GLIDE = 800; // ms
const HOLD = 1500; // ms
const STAGGER = 160; // ms of left-to-right delay inside a glide
const RESUME_AFTER_CLICK = 8000; // ms before autoplay resumes
const GREEN = [16, 163, 127];

function cubicBezier(p1x, p1y, p2x, p2y) {
  const cx = 3 * p1x;
  const bx = 3 * (p2x - p1x) - cx;
  const ax = 1 - cx - bx;
  const cy = 3 * p1y;
  const by = 3 * (p2y - p1y) - cy;
  const ay = 1 - cy - by;
  const sampleX = (t) => ((ax * t + bx) * t + cx) * t;
  const sampleY = (t) => ((ay * t + by) * t + cy) * t;
  const slopeX = (t) => (3 * ax * t + 2 * bx) * t + cx;
  return (x) => {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    let t = x;
    for (let i = 0; i < 8; i += 1) {
      const err = sampleX(t) - x;
      const d = slopeX(t);
      if (Math.abs(err) < 1e-5 || Math.abs(d) < 1e-6) break;
      t -= err / d;
    }
    if (t < 0 || t > 1) {
      let lo = 0;
      let hi = 1;
      t = x;
      for (let i = 0; i < 20; i += 1) {
        if (sampleX(t) < x) lo = t;
        else hi = t;
        t = (lo + hi) / 2;
      }
    }
    return sampleY(t);
  };
}

const ease = cubicBezier(0.45, 0, 0.15, 1);
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

function seeded(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function leafPoint(leaf, u, v) {
  const a = (leaf.a * Math.PI) / 180;
  const dx = Math.cos(a);
  const dy = Math.sin(a);
  const half = leaf.w * Math.sin(Math.PI * u);
  return {
    x: dx * u * leaf.L - dy * v * half,
    y: leaf.yb + dy * u * leaf.L + dx * v * half,
  };
}

function splitCounts(weights, total) {
  const sum = weights.reduce((s, w) => s + w, 0);
  const counts = weights.map((w) => Math.floor((w / sum) * total));
  let rest = total - counts.reduce((s, c) => s + c, 0);
  for (let i = 0; rest > 0; i = (i + 1) % counts.length, rest -= 1) {
    counts[i] += 1;
  }
  return counts;
}

function sortForMorph(points) {
  const maxX = Math.max(...points.map((p) => Math.abs(p.x)), 0.001);
  return points
    .map((p) => ({ ...p, key: p.x / maxX + 0.12 * p.y }))
    .sort((a, b) => a.key - b.key);
}

// Geometry lives in units where y spans [-1, 1] and x spans [-A, A].
function buildGeometry(A) {
  const r = seeded(7);
  const STEM = 22;

  // I — Digitize: an irregular scanned plant.
  const scan = [];
  for (let i = 0; i < STEM; i += 1) {
    const t = i / (STEM - 1);
    scan.push({ x: (r() - 0.5) * 0.05, y: 0.95 - t * 1.3 + (r() - 0.5) * 0.03 });
  }
  const scanLeaves = [
    { yb: 0.55, a: -158, L: 0.7, w: 0.17 },
    { yb: 0.38, a: -24, L: 0.8, w: 0.19 },
    { yb: 0.12, a: -142, L: 0.72, w: 0.16 },
    { yb: -0.05, a: -38, L: 0.68, w: 0.15 },
    { yb: -0.3, a: -97, L: 0.55, w: 0.13 },
  ].map((l) => ({ ...l, a: l.a + (r() - 0.5) * 12, L: l.L * (0.92 + r() * 0.16) }));
  splitCounts(scanLeaves.map((l) => l.L * l.w), N - STEM).forEach((n, li) => {
    for (let k = 0; k < n; k += 1) {
      const p = leafPoint(scanLeaves[li], 0.08 + 0.92 * r(), r() * 2 - 1);
      scan.push({ x: p.x + (r() - 0.5) * 0.02, y: p.y + (r() - 0.5) * 0.02 });
    }
  });

  // II — Understand: points regroup into a mechanism network.
  const span = A * 0.82;
  const centers = [
    [-1, 0.1],
    [-0.55, -0.55],
    [-0.5, 0.58],
    [0, 0],
    [0.5, -0.58],
    [0.55, 0.55],
    [1, -0.08],
  ].map(([f, y]) => ({ x: f * span, y }));
  const edges = [
    [0, 1], [0, 2], [1, 3], [2, 3], [3, 4],
    [3, 5], [4, 6], [5, 6], [1, 4], [2, 5],
  ];
  const network = [];
  splitCounts([1, 1, 1, 1.6, 1, 1, 1], N).forEach((n, ci) => {
    for (let k = 0; k < n; k += 1) {
      const rad = 0.04 + 0.12 * Math.sqrt(r());
      const ang = r() * Math.PI * 2;
      network.push({
        x: centers[ci].x + Math.cos(ang) * rad,
        y: centers[ci].y + Math.sin(ang) * rad,
        c: ci,
      });
    }
  });

  // III — Predict: points flow onto a rising growth curve.
  const X = A * 0.85;
  const curveY = (s) => 0.7 - 1.4 / (1 + Math.exp(-5.5 * s));
  const curve = [];
  const ON = Math.round(N * 0.7);
  for (let i = 0; i < ON; i += 1) {
    const s = -1 + (2 * i) / (ON - 1);
    curve.push({ x: s * X, y: curveY(s) + (r() - 0.5) * 0.015 });
  }
  for (let i = ON; i < N; i += 1) {
    const s = r() * 2 - 1;
    curve.push({ x: s * X, y: curveY(s) + (r() + r() + r() - 1.5) * 0.12 });
  }

  // IV — Design: a clean, symmetric designed plant.
  const design = [];
  for (let i = 0; i < STEM; i += 1) {
    design.push({ x: 0, y: 0.95 - (i / (STEM - 1)) * 1.4 });
  }
  const designLeaves = [
    { yb: 0.5, a: -152, L: 0.74, w: 0.15 },
    { yb: 0.5, a: -28, L: 0.74, w: 0.15 },
    { yb: 0.12, a: -142, L: 0.68, w: 0.14 },
    { yb: 0.12, a: -38, L: 0.68, w: 0.14 },
    { yb: -0.24, a: -128, L: 0.52, w: 0.12 },
    { yb: -0.24, a: -52, L: 0.52, w: 0.12 },
  ];
  splitCounts(designLeaves.map((l) => l.L * l.w), N - STEM).forEach((n, li) => {
    const rows = Math.ceil(n / 3);
    for (let k = 0; k < n; k += 1) {
      const u = (Math.floor(k / 3) + 0.5) / rows;
      const v = [0, 0.98, -0.98][k % 3];
      design.push(leafPoint(designLeaves[li], u, v));
    }
  });

  const shapes = [scan, network, curve, design].map(sortForMorph);
  // Per-target stagger: points on the left start their glide first.
  const delays = shapes.map((shape) => {
    const maxX = Math.max(...shape.map((p) => Math.abs(p.x)), 0.001);
    return shape.map((p) => (STAGGER * (p.x / maxX + 1)) / 2);
  });

  return { A, shapes, delays, centers, edges, X, curveY };
}

function overlayAlpha(k, st, p) {
  if (st.from === st.to) return k === st.to ? 1 : 0;
  if (k === st.from) return clamp01(1 - p / 0.5);
  if (k === st.to) return clamp01((p - 0.5) / 0.5);
  return 0;
}

function draw(ctx, size, geo, st, now) {
  const { W, H, dpr } = size;
  const dark = document.documentElement.dataset.theme === "dark";
  const ink = dark ? [245, 245, 245] : [13, 13, 13];
  const S = H * 0.42;
  const ox = W / 2;
  const oy = H / 2;
  const px = (x) => ox + x * S;
  const py = (y) => oy + y * S;
  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const p = st.from === st.to ? 1 : clamp01((now - st.glideStart) / GLIDE);
  ctx.lineWidth = 1;

  // I — scan line sweeping the plant.
  const a0 = overlayAlpha(0, st, p);
  if (a0 > 0) {
    const sweep = ((now % 2400) / 2400) * 2.1 - 1.05;
    ctx.strokeStyle = rgba(GREEN, 0.4 * a0);
    ctx.beginPath();
    ctx.moveTo(px(-0.95), py(sweep));
    ctx.lineTo(px(0.95), py(sweep));
    ctx.stroke();
  }

  // II — network edges and nodes.
  const a1 = overlayAlpha(1, st, p);
  if (a1 > 0) {
    const { centers, edges } = geo;
    ctx.strokeStyle = rgba(ink, 0.22 * a1);
    ctx.beginPath();
    edges.forEach(([i, j]) => {
      ctx.moveTo(px(centers[i].x), py(centers[i].y));
      ctx.lineTo(px(centers[j].x), py(centers[j].y));
    });
    ctx.stroke();
    ctx.strokeStyle = rgba(ink, 0.07 * a1);
    ctx.beginPath();
    geo.shapes[1].forEach((pt) => {
      ctx.moveTo(px(pt.x), py(pt.y));
      ctx.lineTo(px(centers[pt.c].x), py(centers[pt.c].y));
    });
    ctx.stroke();
  }

  // III — axes and the growth curve.
  const a2 = overlayAlpha(2, st, p);
  if (a2 > 0) {
    const { X, curveY } = geo;
    ctx.strokeStyle = rgba(ink, 0.12 * a2);
    ctx.beginPath();
    ctx.moveTo(px(-X), py(0.84));
    ctx.lineTo(px(X), py(0.84));
    ctx.moveTo(px(-X), py(0.84));
    ctx.lineTo(px(-X), py(-0.84));
    ctx.stroke();
    ctx.strokeStyle = rgba(GREEN, 0.9 * a2);
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i <= 64; i += 1) {
      const s = -1 + (2 * i) / 64;
      const x = px(s * X);
      const y = py(curveY(s));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.lineWidth = 1;
  }

  // IV — dashed axis of symmetry.
  const a3 = overlayAlpha(3, st, p);
  if (a3 > 0) {
    ctx.strokeStyle = rgba(ink, 0.14 * a3);
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(px(0), py(-1));
    ctx.lineTo(px(0), py(1));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Points: colour glides from ink to green as the plant becomes designed.
  const ep = ease(p);
  const c0 = st.from === 3 ? GREEN : ink;
  const c1 = st.to === 3 ? GREEN : ink;
  const col = c0.map((v, i) => Math.round(v + (c1[i] - v) * ep));
  ctx.fillStyle = rgba(col, 0.88);
  const R = W < 520 ? 1.8 : 2.1;
  const from = geo.shapes[st.from];
  const to = geo.shapes[st.to];
  const delay = geo.delays[st.to];
  const elapsed = p * GLIDE;
  ctx.beginPath();
  for (let i = 0; i < N; i += 1) {
    const pi =
      st.from === st.to
        ? 1
        : ease(clamp01((elapsed - delay[i]) / (GLIDE - STAGGER)));
    const x = px(from[i].x + (to[i].x - from[i].x) * pi);
    const y = py(from[i].y + (to[i].y - from[i].y) * pi);
    ctx.moveTo(x + R, y);
    ctx.arc(x, y, R, 0, Math.PI * 2);
  }
  ctx.fill();
}

export default function LayerMorph({ steps, label, evidenceHrefs, evidenceLabel }) {
  const wrapRef = React.useRef(null);
  const canvasRef = React.useRef(null);
  const [active, setActive] = React.useState(0);
  const api = React.useRef({ select: () => {} });

  React.useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return undefined;
    const ctx = canvas.getContext("2d");
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");

    const st = {
      from: 0,
      to: 0,
      glideStart: 0,
      holdStart: performance.now(),
      pausedUntil: 0,
      queued: null,
    };
    const size = { W: 0, H: 0, dpr: 1 };
    let geo = null;
    let raf = 0;
    let visible = false;
    let shown = 0;

    const publish = (k) => {
      if (k !== shown) {
        shown = k;
        setActive(k);
      }
    };

    const render = (now) => {
      if (geo) draw(ctx, size, geo, st, now);
    };

    const resize = () => {
      const W = wrap.clientWidth;
      const H = canvas.clientHeight;
      if (!W || !H) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      size.W = W;
      size.H = H;
      size.dpr = dpr;
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      geo = buildGeometry(W / H);
      render(performance.now());
    };

    const glideTo = (k, now) => {
      if (st.from !== st.to) {
        // Never hand off mid-glide: finish landing first, then go.
        st.queued = k;
        return;
      }
      if (k === st.to) return;
      if (reduced.matches) {
        st.from = k;
        st.to = k;
        st.holdStart = now;
        publish(k);
        render(now);
        return;
      }
      st.to = k;
      st.glideStart = now;
    };

    const tick = (now) => {
      if (st.from !== st.to && now - st.glideStart >= GLIDE) {
        st.from = st.to;
        st.holdStart = now;
        if (st.queued !== null) {
          const q = st.queued;
          st.queued = null;
          glideTo(q, now);
        }
      }
      if (
        st.from === st.to &&
        now >= st.pausedUntil &&
        now - st.holdStart >= HOLD
      ) {
        glideTo((st.to + 1) % 4, now);
      }
      const p = st.from === st.to ? 1 : (now - st.glideStart) / GLIDE;
      publish(p >= 0.5 ? st.to : st.from);
      render(now);
      raf = requestAnimationFrame(tick);
    };

    const start = () => {
      if (raf || reduced.matches || !visible) return;
      raf = requestAnimationFrame(tick);
    };
    const stop = () => {
      cancelAnimationFrame(raf);
      raf = 0;
    };

    api.current.select = (k) => {
      const now = performance.now();
      st.pausedUntil = now + RESUME_AFTER_CLICK;
      glideTo(k, now);
    };

    const ro = new ResizeObserver(resize);
    ro.observe(wrap);
    resize();

    const io = new IntersectionObserver(
      ([entry]) => {
        visible = entry.isIntersecting;
        if (visible) start();
        else stop();
      },
      { threshold: 0.1 }
    );
    io.observe(wrap);

    // Redraw on theme change when not animating (reduced motion / off-screen).
    const mo = new MutationObserver(() => render(performance.now()));
    mo.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    const onMotionChange = () => {
      if (reduced.matches) {
        stop();
        st.from = st.to;
        render(performance.now());
      } else {
        start();
      }
    };
    reduced.addEventListener("change", onMotionChange);

    return () => {
      stop();
      ro.disconnect();
      io.disconnect();
      mo.disconnect();
      reduced.removeEventListener("change", onMotionChange);
    };
  }, []);

  return (
    <div className={styles.morph}>
      <div className={styles.canvasWrap} ref={wrapRef}>
        <canvas className={styles.canvas} ref={canvasRef} aria-hidden="true" />
      </div>
      <ol className={styles.stepper} aria-label={label}>
        {steps.map((step, index) => (
          <li
            key={step}
            className={[
              styles.step,
              index < active ? styles.stepDone : "",
              index === active ? styles.stepActive : "",
            ].join(" ")}
          >
            <button
              type="button"
              className={styles.stepButton}
              onClick={() => api.current.select(index)}
              aria-current={index === active ? "step" : undefined}
            >
              <span className={styles.stepNode} aria-hidden="true">
                {NUMERALS[index]}
              </span>
              <span className={styles.stepLabel}>{step}</span>
            </button>
          </li>
        ))}
      </ol>
      {evidenceHrefs?.[active] && (
        <Link className={styles.evidenceLink} to={evidenceHrefs[active]}>
          {evidenceLabel ? evidenceLabel(steps[active]) : steps[active]}
        </Link>
      )}
    </div>
  );
}
