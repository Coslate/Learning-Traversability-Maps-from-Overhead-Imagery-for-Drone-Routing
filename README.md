# Learning-Traversability-Maps-from-Overhead-Imagery-for-Drone-Routing

A geospatial perception project that converts overhead imagery into planning-facing traversability maps for drone routing.

This project studies how semantic understanding from satellite or aerial imagery can support autonomy-relevant route reasoning. Instead of stopping at pixel-wise land-cover segmentation, the pipeline converts predicted semantic masks into traversability-aware cost maps and evaluates how perception quality affects downstream path planning under urban–rural domain shift.

## Why this project matters

For delivery operations, overhead imagery can provide persistent environmental context before a vehicle enters a region. However, raw imagery is not directly useful to a planner. A routing system needs a structured representation such as:

- semantic regions
- obstacle zones
- traversability or risk costs
- route constraints

This project focuses on the first bridge in that stack:

**overhead imagery → semantic map → traversability map → route planning**

The goal is to make geospatial perception more relevant to autonomy than a standalone segmentation benchmark.

## Project scope

This repository currently focuses on the **map-understanding and planning-facing** part of the pipeline.

### Included
- Semantic segmentation on high-resolution overhead imagery
- Urban/rural domain-aware data loading and evaluation
- Conversion from semantic masks to traversability-aware cost maps
- Route planning on the resulting grid representation
- Analysis of how segmentation errors affect downstream routing quality

### Not yet included
- Online localization from onboard drone imagery
- GPS/INS integration
- Cross-view localization or map matching

To isolate map quality from pose-estimation error, the current routing setup assumes **oracle start/goal coordinates in map space**. This makes the project useful for analyzing whether overhead semantic perception is sufficient to support downstream planning.

## Dataset

The current implementation uses the **LoveDA** land-cover segmentation dataset through **TorchGeo**.

### Why LoveDA
LoveDA is a good fit for this project because it provides:
- high-resolution overhead imagery
- dense semantic segmentation labels
- explicit **urban** and **rural** domains
- a natural testbed for domain shift and downstream robustness

### Semantic classes
- Ignore / no-data
- Background
- Building
- Road
- Water
- Barren
- Forest
- Agriculture

## Method overview

### 1. Semantic map construction
The first stage predicts a dense land-cover segmentation map from overhead RGB imagery.

### 2. Traversability map generation
The semantic map is converted into a planning-facing cost representation. For example, roads and open regions are assigned lower routing cost, while buildings and water are treated as high-cost or invalid traversal regions.

### 3. Route planning
A grid-based planner runs on the generated cost map to estimate feasible routes between known start and goal locations.

### 4. Downstream evaluation
The project evaluates not only segmentation quality, but also routing quality:
- route success rate
- excess path cost
- obstacle-crossing violations
- robustness under urban → rural or rural → urban shift

## Current implementation

The current repository includes:

- LoveDA dataset loading with TorchGeo
- Urban-only, rural-only, or combined-domain dataset construction
- Paired image–mask patch sampling
- Paired augmentations for segmentation
- Baseline visualizations
- Class-distribution analysis
- Domain split summaries

This first stage is intentionally focused on **reliable data loading, observability, and debugging speed** before training larger models.

## Planned next steps

### Baseline modeling
- Train a semantic segmentation baseline such as SegFormer or DeepLabV3+
- Report per-class IoU and mIoU

### Planning-facing evaluation
- Convert segmentation outputs into traversability cost maps
- Run A* routing on the resulting grid
- Measure route success, excess path cost, and failure cases

### Extensions
- Uncertainty-aware routing
- Road skeletonization / graph extraction
- Cross-view localization against overhead maps
- Integration with onboard perception or GPS pose priors

## Repository structure

```text
loveda_project/
├── README.md
├── requirements.txt
├── src/
│   └── loveda_project/
│       ├── __init__.py
│       ├── data.py
│       └── transforms.py
└── scripts/
    └── day1_day2_setup.py
