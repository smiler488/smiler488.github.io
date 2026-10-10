---
slug: brdf-paper
title: Predicting Leaf BRDF from Phenotypic Traits
authors: [liangchao]
category: Plant phenotyping
article_type: Research project
tags: [plant-phenotyping, remote-sensing, machine-learning, crop-modeling]
layers: [UND]
image: /img/brdf_cover.jpg
description: A measuring and analysis framework that predicts leaf BRDF parameters from phenotypic traits in four species, combining directional spectroscopy, Cook–Torrance fitting, ensemble learning and canopy ray tracing (Plant Phenomics, 2025).
---
import AltmetricBadge from '@site/src/components/AltmetricBadge';

## Overview

![Directional spectrum measurement and BRDF prediction workflow](/img/brdf_cover.jpg)

Leaves do not reflect light equally in all directions. Their anatomy, pigments and microscopic surface roughness determine how light is scattered, and therefore how it is distributed inside a canopy, yet most canopy models treat leaves as Lambertian reflectors. Measuring the directional reflectance of every leaf is impractical.

This study develops a framework that predicts leaf optical properties from traits that are easy to measure. It combines a custom-built **Directional Spectrum Detection Instrument (DSDI)**, Cook–Torrance **bidirectional reflectance distribution function (BRDF)** fitting, phenotypic measurements and ensemble learning, and then uses canopy ray tracing to quantify how the predicted optical properties change light distribution in a canopy.

<!-- truncate -->

<AltmetricBadge doi="10.1016/j.plaphe.2025.100135" badgeType="donut" className="brdfAltmetric" />

## At a glance

- **Plant material:** maize, rice, cotton, and poplar leaves from upper and lower canopy positions.
- **Directional spectra:** 400–1000 nm, measured across a broad angular range with the DSDI.
- **BRDF parameters:** roughness $\sigma(\lambda)$, diffuse reflection coefficient $k(\lambda)$, and refractive index $n(\lambda)$.
- **Predictive model:** a stacking ensemble built from support vector, random forest, and gradient boosting regressors.
- **Performance:** BRDF fitting $R^2 > 0.95$; ensemble prediction $R^2 = 0.83$–$0.99$, depending on the parameter.

## Measurement and modeling workflow

### 1. Measure directional reflectance

The DSDI uses a xenon light source, a fiber spectrometer, and mechanically controlled illumination and viewing angles. A Lambertian white reference is used to calibrate reflectance before leaf measurements.

Both adaxial and abaxial surfaces were measured, since the two differ in epidermal structure and optical response.

### 2. Fit the BRDF model

The Cook–Torrance formulation represents diffuse and specular reflection with three wavelength-dependent parameters:

| Parameter | Physical interpretation | Related leaf properties |
| --- | --- | --- |
| $\sigma(\lambda)$ | Microfacet roughness | Epidermal texture and surface irregularity |
| $k(\lambda)$ | Diffuse reflection coefficient | Internal scattering and the diffuse contribution to reflectance |
| $n(\lambda)$ | Refractive index | Refraction and interface reflection, influenced by tissue composition |

Adaptive grid search and least-squares optimization were used to fit these parameters to the measured directional spectra.

### 3. Predict optical parameters from traits

The input variables included leaf thickness, specific leaf weight, pigment measurements, microscopy-derived surface roughness, and wavelength. The stacking model combines:

- Support Vector Regression (SVR)
- Random Forest Regression (RFR)
- Gradient Boosting Regression Trees (GBRT)
- Linear regression as the meta-learner

The model links measured phenotypic traits directly to BRDF parameters, so that leaf optical properties can be estimated without directional spectral measurement.

### 4. Canopy-scale effects

Predicted BRDF parameters were introduced into a rice-canopy ray-tracing workflow based on **fastTracer**. The simulations show that changing roughness, diffuse reflection, or refractive behavior can alter the vertical and angular distribution of light inside a canopy.

## Main findings

1. Directional leaf reflectance can be represented accurately with a physically based BRDF model.
2. Structural and biochemical leaf traits contain useful information for predicting BRDF parameters.
3. Differences in leaf optical properties substantially change the simulated light field inside a canopy, so leaf optics should be represented explicitly rather than assumed uniform.

Together these results connect leaf-scale phenotyping to radiative-transfer and canopy-photosynthesis models.

## Scope and limitations

The model was built from **270 data entries** covering four species, two canopy positions and both leaf surfaces, and applies within the measured trait and wavelength ranges (400–1000 nm); new species and conditions require new measurements to extend it. The canopy simulations quantify changes in light distribution; linking them to field yield is a subsequent step. Extending the dataset to more genotypes, environments, developmental stages and water status is the next step of this work.

## Code and data availability

- [BRDF fitting scripts and Roughness Calculator](https://github.com/PlantSystemsBiology/brdf)
- [fastTracer canopy ray-tracing software](https://github.com/PlantSystemsBiology/fastTracerPublic)
- The study data are available from the corresponding author upon reasonable request, as stated in the published article.

## Citation

Deng, L., Yu, L. X., Mao, L., Wang, Y., Guo, X., Wang, M., Zhang, Y., Song, Q., & Zhu, X.-G. (2025). **Leaf bidirectional reflectance distribution function (BRDF) prediction with phenotypic traits in four species: Development of a novel measuring and analyzing framework.** *Plant Phenomics, 7*(4), 100135. [https://doi.org/10.1016/j.plaphe.2025.100135](https://doi.org/10.1016/j.plaphe.2025.100135)
