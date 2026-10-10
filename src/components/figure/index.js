/**
 * Figure shells (design/DESIGN_SPEC.md §7.1).
 *
 * InteractiveFigure frames an interactive science figure with a number, an
 * "Interactive" label, a caption and a source line (paper DOI, code). The
 * figure body must render a meaningful static state on the server — that
 * server-rendered state is the poster for no-JS readers, print and first
 * paint. Heavy figures can pass `load` (a dynamic import) and `poster`; the
 * module is then fetched only once the figure scrolls into view.
 */
import React from "react";
import Link from "@docusaurus/Link";
import { useIsChinese } from "@site/src/components/ds";
import styles from "./styles.module.css";

const COPY = {
  en: {
    figure: "Figure",
    interactive: "Interactive",
    source: "Source",
    code: "Code",
    load: "Load interactive view",
    loading: "Loading…",
  },
  zh: {
    figure: "图",
    interactive: "交互",
    source: "来源",
    code: "代码",
    load: "加载交互视图",
    loading: "正在加载…",
  },
};

/** True when the user asked the OS for reduced motion. */
export function useReducedMotion() {
  const [reduced, setReduced] = React.useState(false);
  React.useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(query.matches);
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);
  return reduced;
}

function LazyBody({ load, loadProps, poster, posterAlt, labels }) {
  const ref = React.useRef(null);
  const started = React.useRef(false);
  const [Component, setComponent] = React.useState(null);
  const [state, setState] = React.useState("idle"); // idle | loading | failed

  // One entry point for both triggers: scrolling into view or the button.
  const start = React.useCallback(() => {
    if (started.current) return;
    started.current = true;
    setState("loading");
    load()
      .then((mod) => setComponent(() => mod.default))
      .catch(() => {
        started.current = false;
        setState("failed");
      });
  }, [load]);

  React.useEffect(() => {
    const node = ref.current;
    if (!node || typeof IntersectionObserver === "undefined") return undefined;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          observer.disconnect();
          start();
        }
      },
      { rootMargin: "200px" }
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [start]);

  if (Component) return <Component {...loadProps} />;
  return (
    <div ref={ref} className={styles.poster}>
      {poster && <img src={poster} alt={posterAlt ?? ""} />}
      <button
        type="button"
        className={styles.loadButton}
        onClick={start}
        disabled={state === "loading"}
      >
        {state === "loading" ? labels.loading : labels.load}
      </button>
    </div>
  );
}

export function InteractiveFigure({
  number,
  title,
  caption,
  doi,
  code,
  children,
  load,
  loadProps,
  poster,
  posterAlt,
}) {
  const isChinese = useIsChinese();
  const copy = isChinese ? COPY.zh : COPY.en;
  return (
    <figure className={styles.figure}>
      <div className={styles.frame}>
        {load ? (
          <LazyBody
            load={load}
            loadProps={loadProps}
            poster={poster}
            posterAlt={posterAlt}
            labels={copy}
          />
        ) : (
          children
        )}
      </div>
      <figcaption className={styles.caption}>
        <p>
          {number != null && (
            <>
              <strong>
                {copy.figure} {number}.
              </strong>{" "}
            </>
          )}
          <span className={styles.badge}>{copy.interactive}</span> {title}
        </p>
        {caption && <p className={styles.captionText}>{caption}</p>}
        {(doi || code) && (
          <p className={styles.source}>
            {copy.source}:{" "}
            {doi && <Link href={`https://doi.org/${doi}`}>doi:{doi}</Link>}
            {doi && code && " · "}
            {code && <Link href={code}>{copy.code}</Link>}
          </p>
        )}
      </figcaption>
    </figure>
  );
}
