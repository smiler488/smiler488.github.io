import React from "react";
import Layout from "@theme/Layout";
import CitationNotice from "../components/CitationNotice";

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
  ];

  return (
    <Layout title="Digital Plant Phenotyping Platform v25.0">
      <div style={{ textAlign: "center", padding: "40px 20px" }}>
        <h1 style={{ 
          fontSize: "3rem", 
          fontWeight: "700", 
          marginBottom: "16px",
          letterSpacing: "-0.02em",
          color: "#000000"
        }}>
          Digital Plant Phenotyping Platform v25.0
        </h1>
        <p style={{ 
          fontSize: "1.2rem", 
          color: "#666666", 
          marginBottom: "40px",
          maxWidth: "600px",
          margin: "0 auto 40px",
          lineHeight: "1.6"
        }}>
          Select an app to explore.
        </p>

        {/* ✅ App 容器，使用 Grid 布局 */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
            gap: "24px",
            maxWidth: "1200px",
            margin: "0 auto",
            padding: "0 20px",
          }}
        >
          {apps.map((app, index) => (
            <div
              key={index}
              style={{
                border: "1px solid #e5e5e7",
                borderRadius: "16px",
                padding: "24px",
                textAlign: "left",
                backgroundColor: "#ffffff",
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
                minHeight: "200px",
                transition: "all 0.3s ease",
                boxShadow: "0 2px 8px rgba(0, 0, 0, 0.04)",
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.transform = "translateY(-4px)";
                e.currentTarget.style.boxShadow = "0 8px 24px rgba(0, 0, 0, 0.12)";
                e.currentTarget.style.borderColor = "#d2d2d7";
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.transform = "translateY(0)";
                e.currentTarget.style.boxShadow = "0 2px 8px rgba(0, 0, 0, 0.04)";
                e.currentTarget.style.borderColor = "#e5e5e7";
              }}
            >
              <div>
                <h3
                  style={{
                    marginTop: "0",
                    marginBottom: "12px",
                    color: "#000000",
                    fontSize: "1.5rem",
                    fontWeight: "600",
                    letterSpacing: "-0.01em"
                  }}
                >
                  {app.name}
                </h3>
                <p style={{ 
                  flexGrow: 1, 
                  marginBottom: "20px", 
                  overflow: "hidden",
                  color: "#666666",
                  lineHeight: "1.6",
                  fontSize: "1rem"
                }}>
                  {app.description}
                </p>
              </div>
              <a
                href={app.link}
                style={{
                  textDecoration: "none",
                  color: "#ffffff",
                  backgroundColor: "#000000",
                  padding: "12px 20px",
                  borderRadius: "12px",
                  textAlign: "center",
                  fontWeight: "500",
                  transition: "all 0.2s ease",
                  fontSize: "1rem",
                  border: "1px solid #000000"
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.backgroundColor = "#333333";
                  e.currentTarget.style.borderColor = "#333333";
                  e.currentTarget.style.transform = "scale(1.02)";
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.backgroundColor = "#000000";
                  e.currentTarget.style.borderColor = "#000000";
                  e.currentTarget.style.transform = "scale(1)";
                }}
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
