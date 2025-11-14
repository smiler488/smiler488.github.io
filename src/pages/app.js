import React from "react";
import Layout from "@theme/Layout";
import CitationNotice from "../components/CitationNotice";
import styles from "./app.module.css";

const AppHub = () => {
  const apps = [
    {
      name: "Sensor App",
      description:
        "Collect and analyze sensor data from your device in real time, then export complete CSV reports.",
      link: "/app/sensor",
    },
    {
      name: "Land Surveyor",
      description:
        "Plot GPS points manually or via phone location, close the polygon, and instantly estimate surface area.",
      link: "/app/land-survey",
    },
    {
      name: "Irrigation Layout Designer",
      description:
        "Design drip-irrigation-style mainline/submain/drip layouts, visualize SVG hydraulics, and check pressure / flow limits.",
      link: "/app/irrigation-designer",
    },
    {
      name: "Weather Analyzer",
      description:
        "Monitor and analyze climate and agrometeorological data interactively, with clear chart insights.",
      link: "/app/weather",
    },
    {
      name: "Image Quantifier",
      description:
        "Upload or capture leaf images to quantify leaf precisely, export results to CSV.",
      link: "/app/image",
    },
    {
      name: "Root Preprocessor",
      description:
        "Web-native workflow for root scans: polygon ROI selection, high-pass enhancement, and manual cleanup.",
      link: "/app/root-processor",
    },
    {
      name: "Maze App",
      description:
        "Play through a 2D maze simulation with simple physics controls, enjoy an interactive and playful demo.",
      link: "/app/maze",
    },
    {
      name: "CCO Mission Planner",
      description:
        "Upload target area to generate compressed folders with optimized drone CCO mission routes.",
      link: "/app/cco",
    },
    {
      name: "Stereo Rectification",
      description:
        "Rectify stereo video streams in the browser, preview left and right, capture and export PNG or ZIP.",
      link: "/app/stereo",
    },
    {
      name: "Calibration Targets",
      description:
        "Generate printable checkerboards, markers, and AprilTags, download PDF ready to use.",
      link: "/app/targets",
    },
    {
      name: "Cloud Note",
      description:
        "Online encrypted cloud sticky notes, supporting text and files.",
      link: "/app/cloudnote",
    },
    {
      name: "AI Solver",
      description:
        "Take a photo of a problem and get step-by-step solutions powered by Hunyuan AI.",
      link: "/app/solver",
    },
    {
      name: "Journal Selector",
      description:
        "Paste your manuscript abstract and receive journal suggestions plus a downloadable CSV.",
      link: "/app/journal-selector",
    },
    {
      name: "AI Data Visualizer",
      description:
        "Upload CSV/TSV tables and let HunYuan AI summarize trends plus generate interactive ECharts dashboards.",
      link: "/app/ai-data-visualizer",
    },
  ];

  return (
    <Layout title="Digital Plant Phenotyping Platform v25.0">
      <div className={styles.hero}>
        <h1 className={styles.title}>Digital Plant Phenotyping Platform v25.0</h1>
        <p className={styles.subtitle}>Select an app to explore.</p>

        <div className={styles.appGrid}>
          {apps.map((app, index) => (
            <div key={index} className={styles.appCard}>
              <div>
                <h3 className={styles.cardTitle}>{app.name}</h3>
                <p className={styles.cardDescription}>{app.description}</p>
              </div>
              <a
                href={app.link}
                className={styles.appLink}
              >
                Open
              </a>
            </div>
          ))}
        </div>

        <CitationNotice />
      </div>
    </Layout>
  );
};

export default AppHub;
