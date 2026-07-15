const appTutorials = [
  "tutorial-apps/sensor-app-tutorial",
  "tutorial-apps/land-surveyor-tutorial",
  "tutorial-apps/irrigation-layout-designer-tutorial",
  "tutorial-apps/weather-analyzer-tutorial",
  "tutorial-apps/cco-mission-planner-tutorial",
  "tutorial-apps/image-quantifier-tutorial",
  "tutorial-apps/root-preprocessor-tutorial",
  "tutorial-apps/stereo-camera-tutorial",
  "tutorial-apps/calibration-targets-tutorial",
  "tutorial-apps/ai-data-visualizer-tutorial",
  "tutorial-apps/journal-selector-tutorial",
  "tutorial-apps/ai-solver-tutorial",
  "tutorial-apps/cloud-sticky-note-tutorial",
  "tutorial-apps/maze-game-tutorial",
];

const sidebars = {
  tutorialSidebar: [
    {
      type: "category",
      label: "App Lab Tutorials",
      link: {
        type: "generated-index",
        slug: "/category/tutorial---apps",
        title: "App Lab Tutorials",
        description:
          "Verified guides for the 14 browser-based tools in App Lab, including workflows, outputs, privacy boundaries and scientific limitations.",
        keywords: [
          "App Lab",
          "research tools",
          "agriculture",
          "AI for science",
          "tutorials",
        ],
      },
      items: appTutorials,
    },
    {
      type: "category",
      label: "Research Workflows",
      items: ["tutorial-apps/custom-harvard-with-journal-abbr"],
    },
  ],
};

export default sidebars;
