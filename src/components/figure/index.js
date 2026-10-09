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
  },
  zh: { figure: "图", interactive: "交互", source: "来源", code: "代码" },
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

function LazyBody({ load, poster, posterAlt }) {
  const ref = React.useRef(null);
  const [Component, setComponent] = React.useState(null);
  const [failed, setFailed] = React.useState(false);

  React.useEffect(() => {
    const node = ref.current;
    if (!node) return undefined;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          observer.disconnect();
          load()
            .then((mod) => setComponent(() => mod.default))
            .catch(() => setFailed(true));
        }
      },
      { rootMargin: "200px" }
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [load]);

  if (Component && !failed) return <Component />;
  return (
    <div ref={ref} className={styles.poster}>
      {poster && <img src={poster} alt={posterAlt ?? ""} />}
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
  poster,
  posterAlt,
}) {
  const isChinese = useIsChinese();
  const copy = isChinese ? COPY.zh : COPY.en;
  return (
    <figure className={styles.figure}>
      <div className={styles.frame}>
        {load ? (
          <LazyBody load={load} poster={poster} posterAlt={posterAlt} />
        ) : (
          children
        )}
      </div>
      <figcaption className={styles.caption}>
        <p>
          <strong>
            {copy.figure} {number}.
          </strong>{" "}
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
