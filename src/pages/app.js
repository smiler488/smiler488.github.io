import React, { useMemo, useState } from "react";
import Layout from "@theme/Layout";
import Heading from "@theme/Heading";
import Link from "@docusaurus/Link";
import CitationNotice from "../components/CitationNotice";
import { APP_CATEGORIES, APP_MANIFEST } from "../data/appManifest";
import styles from "./app.module.css";

function AppCard({ app }) {
  return (
    <Link
      className={styles.appCard}
      to={app.route}
      data-tone={app.tone}
      aria-label={`Open ${app.name}`}
    >
      <div className={styles.cardTopline}>
        <span className={styles.appIcon} aria-hidden="true">
          {app.icon}
        </span>
        <span className={styles.categoryLabel}>{app.categoryLabel}</span>
      </div>

      <div className={styles.cardCopy}>
        <Heading as="h2" className={styles.cardTitle}>
          {app.shortName}
        </Heading>
        <p className={styles.cardDescription}>{app.description}</p>
      </div>

      <ul className={styles.cardBadges} aria-label="Capabilities">
        {app.badges.slice(0, 3).map((badge) => (
          <li key={badge}>{badge}</li>
        ))}
      </ul>

      <div className={styles.cardAction} aria-hidden="true">
        <span>Open tool</span>
        <span className={styles.cardArrow}>↗</span>
      </div>
    </Link>
  );
}

export default function AppHub() {
  const [activeCategory, setActiveCategory] = useState("all");
  const [query, setQuery] = useState("");

  const visibleApps = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();

    return APP_MANIFEST.filter((app) => {
      const matchesCategory =
        activeCategory === "all" || app.category === activeCategory;
      const searchText = [
        app.name,
        app.shortName,
        app.description,
        app.categoryLabel,
        ...app.badges,
        ...app.keywords,
      ]
        .join(" ")
        .toLowerCase();

      return (
        matchesCategory &&
        (!normalizedQuery || searchText.includes(normalizedQuery))
      );
    });
  }, [activeCategory, query]);

  const clearFilters = () => {
    setActiveCategory("all");
    setQuery("");
  };

  return (
    <Layout
      title="App Lab — Browser Tools for AI and Plant Science"
      description="Fourteen focused browser tools for field data, crop research, imaging, visualization and AI-assisted workflows."
    >
      <main className={styles.page}>
        <div className={styles.ambient} aria-hidden="true">
          <span className={styles.orbitOne} />
          <span className={styles.orbitTwo} />
          <span className={styles.gridGlow} />
        </div>

        <section className={styles.hero} aria-labelledby="app-lab-title">
          <div className={styles.heroCopy}>
            <p className={styles.eyebrow}>
              <span className={styles.liveDot} aria-hidden="true" />
              Digital plant phenotyping platform
            </p>
            <Heading as="h1" className={styles.title} id="app-lab-title">
              A focused lab for field data, imaging and AI.
            </Heading>
            <p className={styles.subtitle}>
              Fourteen practical browser tools shaped around plant science
              workflows. Each workspace now shares one calm, responsive
              interface while keeping its specialist controls close at hand.
            </p>

            <dl className={styles.stats} aria-label="App Lab overview">
              <div>
                <dt>14</dt>
                <dd>browser tools</dd>
              </div>
              <div>
                <dt>5</dt>
                <dd>workflow areas</dd>
              </div>
              <div>
                <dt>BYOK</dt>
                <dd>AI model choice</dd>
              </div>
            </dl>
          </div>

          <div className={styles.heroVisual} aria-hidden="true">
            <div className={styles.visualCore}>
              <span>APP</span>
              <strong>LAB</strong>
            </div>
            <span className={styles.visualNode} data-node="field">
              GPS
            </span>
            <span className={styles.visualNode} data-node="vision">
              CV
            </span>
            <span className={styles.visualNode} data-node="ai">
              AI
            </span>
            <span className={styles.visualNode} data-node="data">
              CSV
            </span>
          </div>
        </section>

        <section className={styles.catalog} aria-labelledby="catalog-title">
          <div className={styles.catalogHeader}>
            <div>
              <p className={styles.sectionEyebrow}>Explore the toolkit</p>
              <Heading as="h2" id="catalog-title">
                Choose a workflow
              </Heading>
              <p>
                Search by task or filter by research stage. Every card opens a
                dedicated workspace.
              </p>
            </div>

            <label className={styles.searchBox}>
              <span className={styles.visuallyHidden}>Search tools</span>
              <span className={styles.searchIcon} aria-hidden="true" />
              <input
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search tools or tasks"
                autoComplete="off"
              />
              {query && (
                <button
                  type="button"
                  onClick={() => setQuery("")}
                  aria-label="Clear search"
                >
                  ×
                </button>
              )}
            </label>
          </div>

          <div className={styles.filterRow}>
            <div
              className={styles.filters}
              aria-label="Filter tools by category"
            >
              {APP_CATEGORIES.map((category) => (
                <button
                  key={category.id}
                  type="button"
                  className={
                    activeCategory === category.id
                      ? styles.filterActive
                      : undefined
                  }
                  aria-pressed={activeCategory === category.id}
                  onClick={() => setActiveCategory(category.id)}
                >
                  {category.label}
                </button>
              ))}
            </div>
            <p className={styles.resultCount} aria-live="polite">
              {visibleApps.length} {visibleApps.length === 1 ? "tool" : "tools"}
            </p>
          </div>

          {visibleApps.length > 0 ? (
            <div className={styles.appGrid}>
              {visibleApps.map((app) => (
                <AppCard key={app.id} app={app} />
              ))}
            </div>
          ) : (
            <div className={styles.emptyState}>
              <span aria-hidden="true">⌕</span>
              <Heading as="h2">No matching tool</Heading>
              <p>Try a broader term or return to the complete App Lab.</p>
              <button type="button" onClick={clearFilters}>
                Show all tools
              </button>
            </div>
          )}
        </section>

        <section
          className={styles.trustStrip}
          aria-label="Privacy and runtime notes"
        >
          <div>
            <strong>Local-first where possible</strong>
            <span>
              Files and calculations stay in the browser unless a tool clearly
              names an external service.
            </span>
          </div>
          <div>
            <strong>Your model, your key</strong>
            <span>
              AI tools use the provider and API credentials you configure for
              that session.
            </span>
          </div>
          <div>
            <strong>Research-aware output</strong>
            <span>
              Scientific assumptions, external runtimes and preliminary
              estimates are labelled in context.
            </span>
          </div>
        </section>

        <div className={styles.citationWrap}>
          <CitationNotice />
        </div>
      </main>
    </Layout>
  );
}
