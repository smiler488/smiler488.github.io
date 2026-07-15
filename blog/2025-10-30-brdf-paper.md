---
slug: brdf-paper
title: Predicting Leaf BRDF from Phenotypic Traits
authors: [liangchao]
category: Plant phenotyping
article_type: Research project
tags: [plant-phenotyping, remote-sensing, machine-learning, crop-modeling]
image: /img/brdf_cover.jpg
description: A peer-reviewed framework combining directional spectroscopy, BRDF fitting, phenotypic traits, ensemble learning, and canopy ray tracing in four species.
---
import AltmetricBadge from '@site/src/components/AltmetricBadge';

## Overview

![Directional spectrum measurement and BRDF prediction workflow](/img/brdf_cover.jpg)

Leaf surfaces do not reflect light uniformly. Their anatomy, pigments, and microscopic roughness change how radiation is scattered through a canopy, yet many canopy models use simplified optical inputs.

This study combines a custom **Directional Spectrum Detection Instrument (DSDI)**, Cook–Torrance **bidirectional reflectance distribution function (BRDF)** fitting, phenotypic measurements, and ensemble learning. The goal is to estimate leaf optical parameters from traits that are easier to measure and then examine how those parameters affect simulated canopy light distribution.

<!-- truncate -->

<AltmetricBadge doi="10.1016/j.plaphe.2025.100135" badgeType="donut" className="brdfAltmetric" />

## At a glance

- **Plant material:** maize, rice, cotton, and poplar leaves from upper and lower canopy positions.
- **Directional spectra:** 400–1000 nm, measured across a broad angular range with the DSDI.
- **BRDF parameters:** roughness $\sigma(\lambda)$, diffuse reflection coefficient $k(\lambda)$, and refractive index $n(\lambda)$.
- **Predictive model:** a stacking ensemble built from support vector, random forest, and gradient boosting regressors.
- **Reported performance:** BRDF fitting $R^2 > 0.95$; ensemble prediction $R^2 = 0.83$–$0.99$, depending on the parameter.

## Measurement and modeling workflow

### 1. Measure directional reflectance

The DSDI uses a xenon light source, a fiber spectrometer, and mechanically controlled illumination and viewing angles. A Lambertian white reference is used to calibrate reflectance before leaf measurements.

Both adaxial and abaxial leaf surfaces were measured. This matters because the two surfaces differ in epidermal structure and optical response.

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

The resulting model provides a direct, data-driven link between measured phenotypic traits and BRDF parameters within the study domain.

### 4. Test canopy-scale consequences

Predicted BRDF parameters were introduced into a rice-canopy ray-tracing workflow based on **fastTracer**. The simulations show that changing roughness, diffuse reflection, or refractive behavior can alter the vertical and angular distribution of light inside a canopy.

## What the results support

The study supports three practical conclusions:

1. Directional leaf reflectance can be represented accurately with a physically based BRDF model.
2. Structural and biochemical leaf traits contain useful information for predicting BRDF parameters.
3. Leaf optical diversity can materially change simulated canopy light fields and should not always be treated as uniform.

These results provide a route for connecting leaf-scale phenotyping to radiative-transfer and canopy-photosynthesis models.

## Scope and limitations

The model was developed from **270 data entries** spanning four species, two canopy positions, and both leaf surfaces. It is therefore a research model, not a universal estimator for every crop, genotype, environment, or stress treatment.

- Predictions outside the measured trait and wavelength ranges require new validation.
- The ray-tracing results demonstrate changes in simulated light distribution; they do not by themselves demonstrate yield gains in the field.
- Direct optical measurement remains important when working with new species or when high-accuracy optical parameters are required.
- Future datasets should cover more genotypes, environments, developmental stages, and water-status conditions.

## Code and data availability

- [BRDF fitting scripts and Roughness Calculator](https://github.com/PlantSystemsBiology/brdf)
- [fastTracer canopy ray-tracing software](https://github.com/PlantSystemsBiology/fastTracerPublic)
- The study data are available from the corresponding author upon reasonable request, as stated in the published article.

## Citation

Deng, L., Yu, L. X., Mao, L., Wang, Y., Guo, X., Wang, M., Zhang, Y., Song, Q., & Zhu, X.-G. (2025). **Leaf bidirectional reflectance distribution function (BRDF) prediction with phenotypic traits in four species: Development of a novel measuring and analyzing framework.** *Plant Phenomics, 7*(4), 100135. [https://doi.org/10.1016/j.plaphe.2025.100135](https://doi.org/10.1016/j.plaphe.2025.100135)
