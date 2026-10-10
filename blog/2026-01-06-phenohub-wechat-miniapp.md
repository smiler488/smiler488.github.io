---
slug: phenohub-wechat-miniapp
title: "PhenoHUB: A Mobile Toolkit for Digital Plant Phenotyping"
authors: [liangchao]
category: Plant phenotyping
article_type: Research project
tags: [plant-phenotyping, artificial-intelligence, data-analysis]
layers: [DIG]
image: /img/phenohub.png
description: "A WeChat Mini Program for plant phenotyping fieldwork: leaf-angle and land-area measurement, site weather, image analysis, AI-assisted charting and data management in one mobile toolkit."
---

## Overview

![PhenoHUB screens](/img/phenohub.png)

Field phenotyping often happens far from a lab computer, yet many routine measurements need only a phone: the angle of a leaf, the area of a plot, the weather at the site, a quick image analysis, a first look at a data table. **PhenoHUB** brings these tools together in one WeChat Mini Program for plant phenotyping and photosynthesis research. It runs on iOS and Android inside WeChat, needs no installation, and connects to a Python backend for the analyses that do not fit on the phone.

<!-- truncate -->

## Modules

### Phenotypic measurement

- **Leaf angle.** The phone's gyroscope and accelerometer give its orientation in real time. With the phone laid along the leaf blade, the reading is the leaf inclination; readings are recorded per leaf, so several leaves of a plant can be measured in sequence.
- **Land area.** The plot boundary is recorded as a GPS track while walking it; the enclosed polygon area is computed and converted between m², ha and mu.
- **Image quantitative analysis.** Plant images are processed with OpenCV to extract leaf area and colour traits, with batch processing of multiple images.

### Environment

- **Agricultural weather.** Location-based weather for the measurement site: temperature, humidity and light conditions, from a weather service, together with agrometeorological indices.
- **Location.** Coordinates and altitude from GPS/BeiDou positioning, with coordinate conversion.

### AI-assisted analysis

- **AI chart agent.** Turns a CSV file into publication-style charts in eight types: bar charts with ANOVA, heatmaps, line charts, histograms, violin plots, scatter plots and radar charts, with checks on the input data before plotting.
- **AI academic assistant.** Supports experimental design, choice of analysis methods and paper writing.

### Data management

CSV and Excel import, data cleaning, descriptive statistics, hypothesis tests and regression, and export of results.

## Architecture

| Layer | Technology |
|---|---|
| Client | Native WeChat Mini Program, TDesign Miniprogram 1.8.6, LESS |
| Charts | Canvas API for lightweight charts, ECharts for Weixin 1.0.2 for interactive charts |
| Backend | Python FastAPI for image analysis, statistics and model calls |

Light interaction and sensor readings stay on the phone; image analysis, statistics and AI functions run on the backend. The project is organized by tool:

```text
PhenoHUB/
├── pages/
│   ├── hub/                         # toolbox home
│   ├── leafAngle/                   # leaf angle
│   ├── landArea/                    # land area
│   ├── agriWeather/                 # agricultural weather
│   ├── imageQuantitativeAnalysis/   # image analysis
│   ├── aiImage/                     # AI chart agent
│   ├── aiJournal/                   # AI academic assistant
│   └── my/                          # user centre
├── components/
├── utils/
└── Backend code/                    # FastAPI service
```

## Applications

- Leaf-angle and plot-area measurement during field surveys of crop architecture.
- Recording site weather alongside phenotypic measurements.
- First-pass image and statistical analysis of experiment data in the field or greenhouse.
- Teaching mobile phenotyping methods.

## Scope and limitations

Phone sensors are not survey or laboratory instruments. Leaf-angle readings depend on how the phone is aligned with the blade, and consumer GPS has metre-level error, so area measurement suits plots and fields rather than small quadrats. Weather values come from the provider's station or grid, not from sensors at the plant. Colour-based image traits are relative unless calibrated against reference measurements. Data sent to the backend and to AI models leave the phone, so sensitive data should be handled accordingly.

## Access

The source code is hosted on WeChat Git with access on request. For access or collaboration, see the contact details on the [CV page](/cv).
