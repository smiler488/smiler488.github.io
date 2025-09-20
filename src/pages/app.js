import React from "react";
import Layout from "@theme/Layout";

const AppHub = () => {
  const apps = [
    {
      name: "Sensor App",
      description:
        "Collect and analyze sensor data from your device in real time, then export complete CSV reports.",
      link: "/app/sensor",
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
      name: "AI Solver (Hunyuan)",
      description:
        "Take a photo of a problem and get step-by-step solutions powered by Hunyuan AI.",
      link: "/app/solver",
    },
  ];

  return (
    <Layout title="App Hub">
      <div style={{ textAlign: "center", padding: "20px" }}>
        <h1>Web App Hub</h1>
        <p>Select an app to explore.</p>

        {/* ✅ App 容器，使用 Flex 布局 */}
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            gap: "20px",
            marginTop: "20px",
            flexWrap: "wrap", // ✅ 允许多行排列
          }}
        >
          {apps.map((app, index) => (
            <div
              key={index}
              style={{
                border: "1px solid #ddd",
                borderRadius: "8px",
                padding: "15px",
                width: "250px",
                background: "#f9f9f9",
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between", // ✅ 确保按钮在底部
                minHeight: "280px",
                maxHeight: "280px",
                boxShadow: "0 4px 8px rgba(0,0,0,0.1)",
                transition: "0.3s", // ✅ hover 过渡效果
              }}
              // ✅ 鼠标悬停时改变背景色
              onMouseOver={(e) => (e.currentTarget.style.backgroundColor = "#f0f8ff")}
              onMouseOut={(e) => (e.currentTarget.style.backgroundColor = "#f9f9f9")}
            >
              <div>
                <h3
                  style={{
                    marginTop: "0",
                    marginBottom: "10px",
                    fontWeight: "bold",
                  }}
                >
                  {app.name}
                </h3>
                <p style={{ flexGrow: 1, marginBottom: "15px", overflow: "hidden" }}>{app.description}</p>
              </div>
              <a
                href={app.link}
                style={{
                  textDecoration: "none",
                  color: "#fff",
                  backgroundColor: "#007BFF",
                  padding: "10px 15px",
                  borderRadius: "5px",
                  textAlign: "center",
                  marginTop: "auto",
                  transition: "0.2s",
                }}
                onMouseOver={(e) => (e.currentTarget.style.backgroundColor = "#0056b3")}
                onMouseOut={(e) => (e.currentTarget.style.backgroundColor = "#007BFF")}
              >
                Open
              </a>
            </div>
          ))}
        </div>
      </div>
    </Layout>
  );
};

export default AppHub;