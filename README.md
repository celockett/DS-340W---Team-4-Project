# DS-340W---Team-4-Project

# AccessAI

AI-Powered Accessibility Detection and Route Recommendation on Penn State's Campus

# Launch the Live AccessAI Application

https://ds-340w---team-4-project-eqnsevs7h4uwqbqoeuerxf.streamlit.app/

## Project Overview

AccessAI is an accessibility-focused navigation project designed to help users of different mobility disability groups identify sidewalk barriers and explore routes across Penn State's campus. The project combines computer vision, manually labeled sidewalk imagery, campus sidewalk network data, and accessibility-aware routing.

The application allows users to select a mobility profile and route preferences, then compares possible paths using accessibility-related information about the sidewalk network.

## Key Features

- Sidewalk accessibility detection: Uses a pretrained DINOv2 vision model with a classification head to identify accessibility-related sidewalk conditions.

- Model evaluation: Compares model predictions against manually assigned labels for 71 Penn State campus sidewalk images.

- Accessibility-aware routing: Uses Dijkstra-based routing algorithms to calculate paths through the campus sidewalk network.

- Mobility personalization: Incorporates mobility profiles, including walking cane, walker, mobility scooter, manual wheelchair, and motorized wheelchair.

- Route preferences: Accounts for accessibility considerations such as avoiding stairs and penalizing sidewalk conditions that may create mobility barriers.

- Weather information: Incorporates current weather information into the application.

## Parent Papers

Towards Fine-Grained Sidewalk Accessibility Assessment with Deep Learning: Initial Benchmarks and an Open Dataset

https://dl.acm.org/doi/10.1145/3663548.3688531

Accessibility for Whom? Perceptions of Mobility Barriers Across Disability Groups and Implications for Designing Personalized Maps

https://dl.acm.org/doi/10.1145/3706598.3713421

Accessibility Routing Model for the Safe Navigation of Vulnerable Road Users in Urban Environments

https://www.sciencedirect.com/science/article/pii/S2667091725000664

## Data

The project uses sidewalk accessibility data and research resources associated with Project Sidewalk and the first parent paper.

Model evaluation uses 71 manually labeled Penn State campus sidewalk images. The labels provide a reference for evaluating predictions across selected sidewalk conditions.

The campus routing component uses sidewalk segment and node information, along with associated accessibility data and imagery.

## Model Evaluation

The DINOv2 classifier was evaluated by comparing its predictions against manual labels for the 71-image dataset.

Evaluation includes:

- ROC-AUC and average precision (AP) for ranking performance.

- Precision, recall, and F1-score at a 0.5 classification threshold.

- ROC and precision-recall curves.

- Comparisons of selected accessibility-related labels and prediction scores.

Evaluation limitations: The datset is relatively small, and some conditions have very few positive examples or no positive examples. Metrics for conditions without positive examples are undefined. Exploratory threshold selection on the same evaluation dataset should not be interpreted as independent validation or evidence of generalization to new images.

## Routing

AccessAI uses Dijkstra-based algorithms to calculate routes through the campus sidewalk network. The routing implementation includes standard routing comparisons and accessibility-focused approaches that incorporate sidewalk conditions and mobility-related preferences.

The project includes route comparisons and network connectivity analysis, including an analysis of network connectivity after excluding stair segments.

## Project Status

AccessAI includes a working application, DINOv2-based sidewalk condition predictions, evaluation results, and accessibility-aware routing implementations. The project demonstrates how computer vision and personalized routing can be combined to support more accessible navigation on Penn State's campus.
