---
slug: phenohub-wechat-miniapp
title: "PhenoHUB: A Mobile Toolkit for Digital Plant Phenotyping"
authors: [liangchao]
category: Plant phenotyping
article_type: Research project
tags: [plant-phenotyping, artificial-intelligence, data-analysis]
image: /img/phenohub.png
description: A January 2026 snapshot of a WeChat Mini Program combining field utilities, weather queries, image tools, and experimental AI assistants.
---

## Overview

![PhenoHUB prototype screens and WeChat Mini Program code](/img/phenohub.png)

**PhenoHUB** is a WeChat Mini Program prototype that groups mobile utilities for plant-phenotyping work. The project explores how field measurements, weather queries, image analysis, and AI-assisted research tasks can be made easier to access from a phone.

This article records the **January 2026 project snapshot** represented by the interface above. It is not a live completeness report, and the presence of a module in the launcher does not by itself establish measurement accuracy or production readiness.

<!-- truncate -->

## Prototype modules

### Device-orientation measurements

The leaf-angle utility uses phone motion sensors to display and record device orientation. It can support rapid relative measurements when the phone is aligned consistently with a leaf.

The result should not be described as a precise leaf angle without a defined mounting method, sensor calibration, reference plane, and validation against a trusted instrument. Device model, case geometry, operator alignment, and motion can all affect the reading.

### Land-area estimation

The area utility records a location track and estimates the enclosed polygon. It is useful for reconnaissance and rough field records, but consumer-phone positioning is not survey-grade.

- Accuracy varies with the device, satellite visibility, buildings, trees, and sampling interval.
- Altitude from a phone location service is especially uncertain.
- Boundaries used for contracts, regulation, engineering, or precision operations require suitable survey equipment.

### Weather and environmental context

The agricultural-weather module is designed to query location-based weather information. Values obtained from a weather service describe the provider's grid or station estimate; they are not direct measurements from the phone.

Soil moisture, canopy temperature, light intensity, or other local variables require an explicit data source or external sensor. The interface should always label the provider, observation or forecast time, units, and location.

### Image analysis

The image module provides an entry point for plant-image preprocessing and quantitative analysis. Any reported area, count, color, or shape trait depends on segmentation quality, scale calibration, and acquisition conditions.

Image color alone should not be presented as chlorophyll content without a documented calibration model and independent validation.

### Experimental AI assistants

The prototype includes entries for chart generation and research assistance. These tools can help explore CSV data or draft analytical ideas, but AI output must remain reviewable:

- Charts should be traced back to the exact input rows and transformations.
- Statistical tests require their assumptions and sample structure to be checked.
- Literature, journal, and writing suggestions can be incomplete or incorrect.
- Research data sent to an external model are subject to that provider's privacy terms.

## Architecture snapshot

The mobile client uses the native WeChat Mini Program environment with reusable UI components and Canvas/ECharts-style visualization. Some analysis tasks can be local to the Mini Program, while network-dependent features can call an external API service.

This split keeps lightweight interaction on the phone but creates clear boundaries:

1. Sensor and location permissions should be requested only when the user starts the relevant tool.
2. Local and remote processing must be identified in the interface.
3. Uploaded files and API requests need size limits, failure states, and privacy guidance.
4. Model-generated results should include the provider, model, and generation time when possible.

## Appropriate use

PhenoHUB is most suitable as a prototype for:

- Field notes and rapid exploratory measurements
- Teaching mobile phenotyping concepts
- Testing interaction designs before instrument integration
- Providing one launch point for small research utilities

It should not be treated as a replacement for calibrated scientific instruments, validated statistical software, or a laboratory data-management system.

## Access and reproducibility

The project repository is hosted on WeChat Git and currently requires authenticated access. External readers cannot reliably clone it from a public URL, so this article does not present public installation or contribution steps.

For an internal or authorized deployment, record at least:

- Mini Program revision and backend revision
- Device model and operating-system version
- Permission and calibration procedure
- Weather, map, or AI provider and request time
- Input files, parameters, raw readings, and exported results

Access questions can be directed through the contact channels on the [CV page](/cv).

## Development priorities

The next useful milestones are evidence-driven rather than percentage-based: validate each measurement against a reference method, label experimental modules clearly, add provenance to exports, document privacy boundaries, and test the complete workflow on both iOS and Android devices.
