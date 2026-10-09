/**
 * Workbench panels shown under every App Lab tool (design/DESIGN_SPEC.md
 * §8, §10): the parameter record of the latest export, next-step tools that
 * accept this tool's outputs, and the local workspace.
 */
import React from "react";
import Link from "@docusaurus/Link";
import Heading from "@theme/Heading";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { APP_MANIFEST, localizeApp } from "@site/src/data/appManifest";
import { DATA_TYPES, MATURITY } from "@site/src/lib/workbench/types";
import {
  EXPORT_EVENT,
  buildProvenance,
} from "@site/src/lib/workbench/provenance";
import {
  WORKSPACE_EVENT,
  clearWorkspace,
  deleteArtifact,
  listArtifacts,
} from "@site/src/lib/workbench/workspace";
import styles from "./styles.module.css";

const COPY = {
  en: {
    record: "Parameter record",
    recordHint:
      "Each export from this tool produces a record of the settings, tool version and file checksums, so the result can be reproduced later.",
    recordEmpty: "No export yet in this session.",
    exported: (n, time) => `${n} file${n === 1 ? "" : "s"} exported at ${time}`,
    download: "Download record",
    copy: "Copy JSON",
    copied: "Copied",
    view: "View record",
    next: "Continue with",
    nextHint: "Tools that accept what this one produces.",
    workspace: "Local workspace",
    workspaceHint:
      "Results handed between tools are kept in this browser only. Clearing site data removes them.",
    remove: "Remove",
    clear: "Clear workspace",
    from: "from",
    version: "Version",
    validated: (d) => `Validated ${d}`,
    notValidated: "Validation not yet recorded",
  },
  zh: {
    record: "参数记录",
    recordHint:
      "本工具的每次导出都会生成一份记录，包含参数设置、工具版本与文件校验值，便于日后复现结果。",
    recordEmpty: "本次会话尚未导出。",
    exported: (n, time) => `${time} 导出了 ${n} 个文件`,
    download: "下载记录",
    copy: "复制 JSON",
    copied: "已复制",
    view: "查看记录",
    next: "下一步可以用",
    nextHint: "能接收本工具输出结果的工具。",
    workspace: "本地工作区",
    workspaceHint:
      "在工具之间传递的结果只保存在当前浏览器中，清除网站数据会将其删除。",
    remove: "移除",
    clear: "清空工作区",
    from: "来自",
    version: "版本",
    validated: (d) => `验证于 ${d}`,
    notValidated: "尚未记录验证日期",
  },
};

function useLocale() {
  const { i18n } = useDocusaurusContext();
  return i18n.currentLocale === "zh-Hans" ? "zh" : "en";
}

/** Maturity pill: stable (ink), beta (grey), experimental (amber outline). */
export function MaturityBadge({ level }) {
  const locale = useLocale();
  const info = MATURITY[level];
  if (!info) return null;
  return (
    <span
      className={`${styles.maturity} ${styles[`maturity_${level}`]}`}
      title={info.hint[locale]}
    >
      {info[locale]}
    </span>
  );
}

/** Maturity, version and validation status, shown under a tool's title. */
export function ToolStatus({ app }) {
  const locale = useLocale();
  const copy = COPY[locale];
  if (!app?.maturity) return null;
  return (
    <div className={styles.status}>
      <MaturityBadge level={app.maturity} />
      <span>
        {copy.version} {app.version}
      </span>
      <span aria-hidden="true">·</span>
      <span>
        {app.validatedAt ? copy.validated(app.validatedAt) : copy.notValidated}
      </span>
    </div>
  );
}

function ParameterRecord({ app }) {
  const locale = useLocale();
  const copy = COPY[locale];
  const { siteConfig } = useDocusaurusContext();
  const [record, setRecord] = React.useState(null);
  const [copied, setCopied] = React.useState(false);

  React.useEffect(() => {
    let alive = true;
    async function onExport(event) {
      const next = await buildProvenance({
        app,
        detail: event.detail ?? {},
        build: siteConfig.customFields?.build,
        siteUrl: siteConfig.url,
      });
      if (alive) setRecord(next);
    }
    window.addEventListener(EXPORT_EVENT, onExport);
    return () => {
      alive = false;
      window.removeEventListener(EXPORT_EVENT, onExport);
    };
  }, [app, siteConfig]);

  const json = record ? JSON.stringify(record, null, 2) : "";
  const baseName =
    record?.outputs?.[0]?.name?.replace(/\.[^.]+$/, "") || app.id;

  function download() {
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${baseName}.provenance.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  async function copyJson() {
    try {
      await navigator.clipboard.writeText(json);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  }

  const time = record
    ? new Date(record.createdAt).toLocaleTimeString(
        locale === "zh" ? "zh-CN" : "en",
        {
          hour: "2-digit",
          minute: "2-digit",
        }
      )
    : null;

  return (
    <section className={styles.panel} aria-labelledby={`${app.id}-record`}>
      <div className={styles.panelHead}>
        <Heading as="h2" id={`${app.id}-record`}>
          {copy.record}
        </Heading>
        <p>{copy.recordHint}</p>
      </div>
      {record ? (
        <div className={styles.recordBody} aria-live="polite">
          <p className={styles.recordSummary}>
            {copy.exported(record.outputs.length, time)}
          </p>
          <div className={styles.actions}>
            <button type="button" className={styles.primary} onClick={download}>
              {copy.download}
            </button>
            <button
              type="button"
              className={styles.secondary}
              onClick={copyJson}
            >
              {copied ? copy.copied : copy.copy}
            </button>
          </div>
          <details className={styles.details}>
            <summary>{copy.view}</summary>
            <pre>{json}</pre>
          </details>
        </div>
      ) : (
        <p className={styles.muted}>{copy.recordEmpty}</p>
      )}
    </section>
  );
}

function NextSteps({ app }) {
  const locale = useLocale();
  const copy = COPY[locale];
  const outputs = app.outputs ?? [];
  const next = APP_MANIFEST.filter(
    (other) =>
      other.id !== app.id &&
      other.inputs?.some((type) => outputs.includes(type))
  );
  if (!next.length) return null;
  return (
    <section className={styles.panel} aria-labelledby={`${app.id}-next`}>
      <div className={styles.panelHead}>
        <Heading as="h2" id={`${app.id}-next`}>
          {copy.next}
        </Heading>
        <p>{copy.nextHint}</p>
      </div>
      <ul className={styles.nextList}>
        {next.map((other) => {
          const shared = other.inputs.filter((t) => outputs.includes(t));
          return (
            <li key={other.id}>
              <Link to={other.route}>
                <strong>{localizeApp(other.name, locale === "zh")}</strong>
                <span>
                  {shared.map((t) => DATA_TYPES[t]?.[locale] ?? t).join(" · ")}
                </span>
              </Link>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function Workspace() {
  const locale = useLocale();
  const copy = COPY[locale];
  const [items, setItems] = React.useState([]);

  React.useEffect(() => {
    let alive = true;
    const load = () =>
      listArtifacts()
        .then((list) => alive && setItems(list))
        .catch(() => alive && setItems([]));
    load();
    window.addEventListener(WORKSPACE_EVENT, load);
    return () => {
      alive = false;
      window.removeEventListener(WORKSPACE_EVENT, load);
    };
  }, []);

  if (!items.length) return null;
  const nameOf = (appId) => {
    const app = APP_MANIFEST.find((a) => a.id === appId);
    return app
      ? localizeApp(app.shortName ?? app.name, locale === "zh")
      : appId;
  };
  const format = (iso) =>
    new Date(iso).toLocaleString(locale === "zh" ? "zh-CN" : "en", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });

  return (
    <section className={styles.panel} aria-labelledby="lab-workspace">
      <div className={styles.panelHead}>
        <Heading as="h2" id="lab-workspace">
          {copy.workspace}
        </Heading>
        <p>{copy.workspaceHint}</p>
      </div>
      <ul className={styles.workspaceList}>
        {items.map((item) => (
          <li key={item.id}>
            <div>
              <strong>{item.label}</strong>
              <span>
                {DATA_TYPES[item.type]?.[locale] ?? item.type} · {copy.from}{" "}
                {nameOf(item.appId)} · {format(item.createdAt)}
              </span>
            </div>
            <button
              type="button"
              className={styles.secondary}
              onClick={() => deleteArtifact(item.id)}
            >
              {copy.remove}
            </button>
          </li>
        ))}
      </ul>
      <button
        type="button"
        className={styles.textButton}
        onClick={() => clearWorkspace()}
      >
        {copy.clear}
      </button>
    </section>
  );
}

export default function WorkbenchPanels({ app }) {
  if (!app) return null;
  return (
    <div className={styles.panels}>
      {app.provenance && <ParameterRecord app={app} />}
      <NextSteps app={app} />
      <Workspace />
    </div>
  );
}
