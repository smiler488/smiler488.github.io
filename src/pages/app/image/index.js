import React, { useEffect, useState } from "react";
import Link from "@docusaurus/Link";
import { useColorMode } from "@docusaurus/theme-common";

import AppScaffold from "../../../components/AppScaffold";
import CitationNotice from "../../../components/CitationNotice";
import styles from "./styles.module.css";
import { makeToolText } from "@site/src/lib/i18n/toolText";
import ZH from "./_zh";

const tx = makeToolText(ZH);

const SPACE_URL = "https://smiler488-image-quantifier.hf.space";
const SPACE_API_URL =
  "https://huggingface.co/api/spaces/smiler488/image-quantifier";

function describeStage(stage) {
  const value = String(stage || "").toUpperCase();
  if (["RUNNING", "READY"].includes(value)) {
    return { tone: "ready", label: tx("Hosted runtime ready") };
  }
  if (["APP_STARTING", "BUILDING", "STARTING", "SLEEPING"].includes(value)) {
    return { tone: "starting", label: tx("Hosted runtime is waking up") };
  }
  if (["RUNTIME_ERROR", "BUILD_ERROR", "STOPPED", "PAUSED"].includes(value)) {
    return { tone: "error", label: tx("Hosted runtime is unavailable") };
  }
  return { tone: "checking", label: tx("Checking hosted runtime") };
}

function BiologicalWorkspace() {
  const { colorMode } = useColorMode();
  const [retryKey, setRetryKey] = useState(0);
  const [service, setService] = useState({
    tone: "checking",
    label: tx("Checking hosted runtime"),
  });
  const [frameLoaded, setFrameLoaded] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 8000);

    fetch(SPACE_API_URL, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(tx("Status {0}", response.status));
        return response.json();
      })
      .then((payload) => setService(describeStage(payload?.runtime?.stage)))
      .catch(() => {
        if (!controller.signal.aborted) {
          setService({
            tone: "unknown",
            label: tx("Runtime status could not be checked"),
          });
        }
      })
      .finally(() => window.clearTimeout(timeoutId));

    return () => {
      window.clearTimeout(timeoutId);
      controller.abort();
    };
  }, [retryKey]);

  const frameUrl = `${SPACE_URL}/?__theme=${
    colorMode === "dark" ? "dark" : "light"
  }`;

  return (
    <div className={styles.workspace}>
      <div className={styles.serviceBar} role="status" aria-live="polite">
        <span
          className={`${styles.statusDot} ${
            styles[`statusDot--${service.tone}`]
          }`}
          aria-hidden="true"
        />
        <div>
          <strong>
            {frameLoaded ? tx("Analysis workspace loaded") : service.label}
          </strong>
          <span>
            {tx(
              "The embedded app runs on Hugging Face infrastructure; startup can take a minute after inactivity."
            )}
          </span>
        </div>
        <div className={styles.serviceActions}>
          <button
            type="button"
            className="button button--secondary"
            onClick={() => {
              setService({
                tone: "checking",
                label: tx("Checking hosted runtime"),
              });
              setFrameLoaded(false);
              setRetryKey((value) => value + 1);
            }}
          >
            {tx("Retry")}
          </button>
          <Link
            className="button button--secondary"
            to={SPACE_URL}
            target="_blank"
            rel="noreferrer"
          >
            {tx("Open separately")}
          </Link>
        </div>
      </div>

      <aside className={styles.privacyNotice} role="note">
        {tx(
          "Images selected inside the embedded workspace are processed by the hosted Space, not by this static website. Do not upload sensitive material without reviewing the hosted service first."
        )}
      </aside>

      <section
        className={styles.frameCard}
        aria-label={tx("Hosted biological sample analysis workspace")}
      >
        {!frameLoaded && (
          <div className={styles.loadingLayer} aria-hidden="true">
            <span className={styles.spinner} />
            <span>{tx("Loading analysis workspace…")}</span>
          </div>
        )}
        <iframe
          key={`${retryKey}-${colorMode}`}
          src={frameUrl}
          title={tx("Biological Sample Quantifier hosted workspace")}
          className={styles.frame}
          allow="camera; clipboard-write"
          referrerPolicy="strict-origin-when-cross-origin"
          loading="lazy"
          onLoad={() => setFrameLoaded(true)}
        />
      </section>

      <CitationNotice />
    </div>
  );
}

export default function BiologicalSampleAnalysisApp() {
  return (
    <AppScaffold appId="image">
      <BiologicalWorkspace />
    </AppScaffold>
  );
}
