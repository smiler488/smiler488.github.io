/**
 * three.js viewer for the cotton 3D datasets (lazy-loaded; see index.js).
 *
 * mode "sfm"     single SfM point cloud, coloured by organ class
 * mode "compare" SfM vs Hunyuan3D (same aligned frame): either cloud by
 *                class, or both overlaid and coloured by source
 * mode "canopy"  triangle-facet canopy mesh with leaf / non-leaf classes
 *
 * Renders on demand (only while the view changes), handles resize and theme
 * changes, and releases GPU resources on unmount.
 */
import React from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import useBaseUrl from "@docusaurus/useBaseUrl";
import { COTTON3D, POINT_CLASSES } from "@site/src/data/cotton3d";
import { useIsChinese, useLocalize } from "@site/src/components/ds";
import styles from "./styles.module.css";

// Okabe–Ito colours for organ classes (data palette, DESIGN_SPEC §7.4).
const CLASS_COLORS = { 0: "#D55E00", 1: "#E69F00", 2: "#009E73" };
const SOURCE_COLORS = {
  sfm: { light: "#1a1a1a", dark: "#e5e5e5" },
  hy3d: "#0072B2",
};
const LEAF_COLOR = "#3f9a62";
const NONLEAF_COLOR = "#8a6a4a";

const COPY = {
  en: {
    loading: "Loading 3D data…",
    failed:
      "3D view unavailable in this browser. The image above shows the same data.",
    reset: "Reset view",
    sfm: "SfM",
    hy3d: "Hunyuan3D",
    overlay: "Overlay",
    showNonleaf: "Non-leaf organs",
    showEdges: "Triangle edges",
    leaf: "Leaves",
    nonleaf: "Stems and other organs",
    hint: "Drag to rotate · scroll or pinch to zoom · right-drag to pan",
    points: (n) => `${n.toLocaleString("en")} points`,
    triangles: (n) => `${n.toLocaleString("en")} triangles shown`,
    aria: {
      sfm: "Rotatable 3D point cloud of a cotton plant reconstructed with structure from motion.",
      compare:
        "Rotatable 3D comparison of SfM and Hunyuan3D point clouds of the same cotton plant.",
      canopy: "Rotatable 3D triangle-facet model of a 24-plant cotton canopy.",
    },
  },
  zh: {
    loading: "正在加载三维数据…",
    failed: "当前浏览器无法显示三维视图，上方图片展示的是同一份数据。",
    reset: "重置视角",
    sfm: "SfM",
    hy3d: "Hunyuan3D",
    overlay: "叠加",
    showNonleaf: "非叶器官",
    showEdges: "三角面元边线",
    leaf: "叶片",
    nonleaf: "茎及其他器官",
    hint: "拖动旋转 · 滚轮或双指缩放 · 右键拖动平移",
    points: (n) => `${n.toLocaleString("zh-CN")} 个点`,
    triangles: (n) => `显示 ${n.toLocaleString("zh-CN")} 个三角面元`,
    aria: {
      sfm: "可旋转的棉花单株 SfM 重建三维点云。",
      compare: "同一棉花单株的 SfM 与 Hunyuan3D 点云三维对比，可旋转。",
      canopy: "可旋转的 24 株棉花冠层三角面元模型。",
    },
  },
};

function isDark() {
  return document.documentElement.getAttribute("data-theme") === "dark";
}

function dequantize(u16, min, max, count, offset = 0) {
  const out = new Float32Array(count * 3);
  for (let i = 0; i < count; i += 1) {
    for (let a = 0; a < 3; a += 1) {
      out[i * 3 + a] =
        min[a] + (u16[offset + i * 3 + a] / 65535) * (max[a] - min[a]);
    }
  }
  return out;
}

async function loadPoints(url, meta) {
  const buffer = await (await fetch(url)).arrayBuffer();
  const n = meta.count;
  const positions = dequantize(
    new Uint16Array(buffer, 0, n * 3),
    meta.min,
    meta.max,
    n
  );
  const labels = new Uint8Array(buffer, n * 6, n);
  return { positions, labels };
}

async function loadCanopy(url, meta) {
  const buffer = await (await fetch(url)).arrayBuffer();
  const nl = meta.leaf.vertices;
  const nn = meta.nonleaf.vertices;
  const u16 = new Uint16Array(buffer);
  const leafPos = dequantize(u16, meta.min, meta.max, nl, 0);
  const nonPos = dequantize(u16, meta.min, meta.max, nn, nl * 3);
  const idxStart = (nl + nn) * 3;
  const leafIdx = u16.slice(idxStart, idxStart + meta.leaf.triangles * 3);
  const nonIdx = u16.slice(
    idxStart + meta.leaf.triangles * 3,
    idxStart + (meta.leaf.triangles + meta.nonleaf.triangles) * 3
  );
  return { leafPos, nonPos, leafIdx, nonIdx };
}

function colorsByClass(labels) {
  const colors = new Float32Array(labels.length * 3);
  const palette = Object.fromEntries(
    Object.entries(CLASS_COLORS).map(([k, hex]) => [
      k,
      new THREE.Color(hex).convertSRGBToLinear(),
    ])
  );
  labels.forEach((label, i) => {
    const c = palette[label] ?? palette[2];
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  });
  return colors;
}

export default function Cotton3DViewer({ mode }) {
  const isChinese = useIsChinese();
  const localize = useLocalize();
  const copy = isChinese ? COPY.zh : COPY.en;
  const mountRef = React.useRef(null);
  const sceneRef = React.useRef(null);
  const [status, setStatus] = React.useState("loading");
  const [view, setView] = React.useState(
    mode === "compare" ? "overlay" : "single"
  );
  const [showNonleaf, setShowNonleaf] = React.useState(true);
  const [showEdges, setShowEdges] = React.useState(false);
  const urls = {
    sfm: useBaseUrl("/data/cotton3d/sfm.bin"),
    hy3d: useBaseUrl("/data/cotton3d/hy3d.bin"),
    canopy: useBaseUrl("/data/cotton3d/canopy.bin"),
  };

  // Build the scene once per mode.
  React.useEffect(() => {
    const mount = mountRef.current;
    let disposed = false;
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      setStatus("failed");
      return undefined;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    mount.appendChild(renderer.domElement);
    renderer.domElement.setAttribute("role", "img");
    renderer.domElement.setAttribute("aria-label", copy.aria[mode]);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 0.005, 50);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;

    let frames = 0;
    let rafId = null;
    // controls.update() can emit "change", which calls requestRender; rafId
    // stays set until the frame is done so that call only extends `frames`.
    const loop = () => {
      const moved = controls.update();
      renderer.render(scene, camera);
      rafId = null;
      if (moved || frames > 0) {
        frames = Math.max(0, frames - 1);
        rafId = requestAnimationFrame(loop);
      }
    };
    // Paint now if idle, then keep animating for n frames (damping).
    const requestRender = (n = 2) => {
      frames = Math.max(frames, n);
      if (rafId != null) return;
      renderer.render(scene, camera);
      rafId = requestAnimationFrame(loop);
    };
    controls.addEventListener("change", () => requestRender(2));
    controls.addEventListener("end", () => requestRender(30));

    const resize = () => {
      const { clientWidth: w, clientHeight: h } = mount;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      requestRender(1);
    };
    const ro = new ResizeObserver(resize);
    ro.observe(mount);

    const frame = (box) => {
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const radius = size.length() / 2;
      const dist =
        (radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2))) * 0.9;
      const dir = new THREE.Vector3(0.9, 0.45, 1.25).normalize();
      camera.position.copy(center).addScaledVector(dir, dist);
      camera.near = dist / 100;
      camera.far = dist * 10;
      camera.updateProjectionMatrix();
      controls.target.copy(center);
      controls.saveState();
      controls.update();
    };

    const objects = {};
    const build = async () => {
      if (mode === "canopy") {
        const meta = COTTON3D.canopy;
        const data = await loadCanopy(urls.canopy, meta);
        if (disposed) return;
        scene.add(new THREE.HemisphereLight(0xffffff, 0x6b5b4b, 1.4));
        const sun = new THREE.DirectionalLight(0xffffff, 1.6);
        sun.position.set(0.6, 1.2, 0.8);
        scene.add(sun);
        const makeMesh = (pos, idx, color) => {
          const g = new THREE.BufferGeometry();
          g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
          g.setIndex(new THREE.BufferAttribute(idx, 1));
          g.computeVertexNormals();
          const m = new THREE.MeshLambertMaterial({
            color,
            side: THREE.DoubleSide,
            flatShading: true,
          });
          const mesh = new THREE.Mesh(g, m);
          const edges = new THREE.Mesh(
            g,
            new THREE.MeshBasicMaterial({
              color: 0x000000,
              wireframe: true,
              transparent: true,
              opacity: 0.18,
            })
          );
          edges.visible = false;
          scene.add(mesh, edges);
          return { mesh, edges };
        };
        objects.leaf = makeMesh(data.leafPos, data.leafIdx, LEAF_COLOR);
        objects.nonleaf = makeMesh(data.nonPos, data.nonIdx, NONLEAF_COLOR);
        const box = new THREE.Box3()
          .setFromObject(objects.leaf.mesh)
          .union(new THREE.Box3().setFromObject(objects.nonleaf.mesh));
        frame(box);
      } else {
        const want = mode === "compare" ? ["sfm", "hy3d"] : ["sfm"];
        const loaded = await Promise.all(
          want.map((id) => loadPoints(urls[id], COTTON3D[id]))
        );
        if (disposed) return;
        const box = new THREE.Box3();
        want.forEach((id, i) => {
          const { positions, labels } = loaded[i];
          const g = new THREE.BufferGeometry();
          g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
          g.setAttribute(
            "color",
            new THREE.BufferAttribute(colorsByClass(labels), 3)
          );
          g.userData.classColors = g.getAttribute("color").array.slice();
          const material = new THREE.PointsMaterial({
            size: id === "hy3d" ? 0.0022 : 0.0028,
            sizeAttenuation: true,
            vertexColors: true,
          });
          const points = new THREE.Points(g, material);
          scene.add(points);
          objects[id] = points;
          g.computeBoundingBox();
          box.union(g.boundingBox);
        });
        frame(box);
      }
      sceneRef.current = { objects, requestRender, controls };
      setStatus("ready");
      resize();
      requestRender(2);
    };
    build().catch(() => !disposed && setStatus("failed"));

    const themeObserver = new MutationObserver(() => {
      sceneRef.current?.applyView?.();
      requestRender(1);
    });
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    return () => {
      disposed = true;
      themeObserver.disconnect();
      ro.disconnect();
      if (rafId != null) cancelAnimationFrame(rafId);
      controls.dispose();
      scene.traverse((obj) => {
        obj.geometry?.dispose?.();
        obj.material?.dispose?.();
      });
      renderer.dispose();
      renderer.domElement.remove();
      sceneRef.current = null;
    };
    // Scene is rebuilt only when the mode changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  // Apply view toggles without rebuilding the scene.
  React.useEffect(() => {
    const s = sceneRef.current;
    if (!s || status !== "ready") return;
    const apply = () => {
      const { objects } = s;
      if (mode === "canopy") {
        objects.nonleaf.mesh.visible = showNonleaf;
        objects.leaf.edges.visible = showEdges;
        objects.nonleaf.edges.visible = showEdges && showNonleaf;
      } else if (mode === "compare") {
        objects.sfm.visible = view !== "hy3d";
        objects.hy3d.visible = view !== "sfm";
        ["sfm", "hy3d"].forEach((id) => {
          const attr = objects[id].geometry.getAttribute("color");
          if (view === "overlay") {
            const hex =
              id === "sfm"
                ? SOURCE_COLORS.sfm[isDark() ? "dark" : "light"]
                : SOURCE_COLORS.hy3d;
            const c = new THREE.Color(hex).convertSRGBToLinear();
            for (let i = 0; i < attr.count; i += 1)
              attr.setXYZ(i, c.r, c.g, c.b);
          } else {
            attr.array.set(objects[id].geometry.userData.classColors);
          }
          attr.needsUpdate = true;
        });
      }
      s.requestRender(2);
    };
    s.applyView = apply;
    apply();
  }, [mode, view, showNonleaf, showEdges, status]);

  const reset = () => {
    const s = sceneRef.current;
    if (!s) return;
    s.controls.reset();
    s.requestRender(2);
  };

  const legend =
    mode === "canopy"
      ? [
          { color: LEAF_COLOR, label: copy.leaf },
          { color: NONLEAF_COLOR, label: copy.nonleaf },
        ]
      : view === "overlay" && mode === "compare"
      ? [
          { color: "var(--ifm-color-emphasis-900)", label: copy.sfm },
          { color: SOURCE_COLORS.hy3d, label: copy.hy3d },
        ]
      : Object.entries(POINT_CLASSES).map(([k, name]) => ({
          color: CLASS_COLORS[k],
          label: localize(name),
        }));

  const count =
    mode === "canopy"
      ? copy.triangles(
          COTTON3D.canopy.leaf.triangles +
            (showNonleaf ? COTTON3D.canopy.nonleaf.triangles : 0)
        )
      : mode === "compare" && view === "hy3d"
      ? copy.points(COTTON3D.hy3d.count)
      : mode === "compare" && view === "overlay"
      ? `${copy.points(COTTON3D.sfm.count)} + ${copy.points(
          COTTON3D.hy3d.count
        )}`
      : copy.points(COTTON3D.sfm.count);

  return (
    <div className={styles.viewer}>
      <div className={styles.toolbar}>
        {mode === "compare" && (
          <div className={styles.segmented} role="group">
            {["sfm", "hy3d", "overlay"].map((id) => (
              <button
                key={id}
                type="button"
                aria-pressed={view === id}
                className={view === id ? styles.segmentActive : styles.segment}
                onClick={() => setView(id)}
              >
                {copy[id]}
              </button>
            ))}
          </div>
        )}
        {mode === "canopy" && (
          <div className={styles.toggles}>
            <label>
              <input
                type="checkbox"
                checked={showNonleaf}
                onChange={(e) => setShowNonleaf(e.target.checked)}
              />
              {copy.showNonleaf}
            </label>
            <label>
              <input
                type="checkbox"
                checked={showEdges}
                onChange={(e) => setShowEdges(e.target.checked)}
              />
              {copy.showEdges}
            </label>
          </div>
        )}
        <button
          type="button"
          className={styles.reset}
          onClick={reset}
          disabled={status !== "ready"}
        >
          {copy.reset}
        </button>
      </div>

      <div className={styles.stage} ref={mountRef}>
        {status !== "ready" && (
          <p className={styles.status} role="status">
            {status === "failed" ? copy.failed : copy.loading}
          </p>
        )}
      </div>

      <div className={styles.footer}>
        <ul className={styles.legend}>
          {legend.map((item) => (
            <li key={item.label}>
              <span
                className={styles.swatch}
                style={{ background: item.color }}
                aria-hidden="true"
              />
              {item.label}
            </li>
          ))}
        </ul>
        <span className={styles.meta}>
          {count} · {copy.hint}
        </span>
      </div>
    </div>
  );
}
