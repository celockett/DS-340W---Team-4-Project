# DS-340W---Team-4-Project

# AccessAI

AI-Powered Accessibility Detection and Route Recommendation on Penn State's Campus

## Project Description

AccessAI aims to explore accessibility-aware route recommendation on Penn State's campus. Our project will build from existing research in sidewalk accessibility detection and routing.

## Parent Papers

Towards Fine-Grained Sidewalk Accessibility Assessment with Deep Learning: Initial Benchmarks and an Open Dataset

https://dl.acm.org/doi/10.1145/3663548.3688531

Accessibility for Whom? Perceptions of Mobility Barriers Across Disability Groups and Implications for Designing Personalized Maps

https://dl.acm.org/doi/10.1145/3706598.3713421

Accessibility Routing Model for the Safe Navigation of Vulnerable Road Users in Urban Environments

https://www.sciencedirect.com/science/article/pii/S2667091725000664

## Dataset

We use sidewalk accessibility data and research resources associated with Project Sidewalk and the first parent paper. Our current preliminary evaluation also uses 68 campus sidewalk images with manual labels to examine model predictions for selected accessibility-related conditions.

## Current Direction

Our work currently focuses on:

- Applying a pretrained DINOv2 vision model to sidewalk images.

- Evaluating predictions for accessibility-related sidewalk conditions against manual labels.

- Comparing model performance across conditions using evaluation metrics and plots.

## Project Status

This project is a work in progress. Current model evaluation is preliminary, and the proposed routing and personalization components are not yet represented as a completed end-to-end system.
