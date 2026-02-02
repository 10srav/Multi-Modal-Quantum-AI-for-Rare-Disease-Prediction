---
created: 2026-02-02T23:02
title: Connect multi-modal fusion to prediction pipeline
area: api
files:
  - src/features/fusion.py
  - dashboard.py
  - api.py
---

## Problem

The system was designed to combine facial features with clinical data (multi-modal fusion), but currently only clinical inputs are used. The fusion module exists in code (LateFusionClassifier, MultimodalFusionModel, GatedFusion, AttentionFusion) but isn't connected to the actual prediction pipeline.

The dashboard and API only use tabular data for predictions, completely bypassing the facial feature extraction (FaceCNN) and fusion layers that are the core differentiator of this project.

## Solution

1. Update dashboard to accept image uploads alongside clinical data
2. Wire the LateFusionClassifier or MultimodalFusionModel into the prediction endpoint
3. Load pre-trained fusion model weights in the API
4. Ensure image preprocessing (224x224, ImageNet normalization) is applied before inference
