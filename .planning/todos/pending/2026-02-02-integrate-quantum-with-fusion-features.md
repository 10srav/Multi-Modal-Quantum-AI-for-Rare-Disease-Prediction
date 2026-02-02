---
created: 2026-02-02T23:02
title: Integrate quantum models with fused embeddings
area: qml
files:
  - src/qml/qsvm.py
  - src/qml/qnn.py
  - src/features/fusion.py
  - src/qml/quantum_features.py
---

## Problem

Even though the backend API supports quantum inference, the dashboard never calls those endpoints. More critically, the quantum models only use simple tabular data — they don't incorporate the facial features or fused embeddings from the multi-modal fusion module.

This means the quantum advantage (if any) is only tested on 11 tabular features, not on the richer fused representation that combines facial + clinical data.

## Solution

1. Extract fused embeddings from MultimodalFusionModel
2. Reduce embedding dimension for quantum encoding (PCA or learned projection)
3. Scale to [0, 2*pi] for quantum feature maps
4. Train QSVM/QNN on fused embeddings instead of raw tabular
5. Add API endpoints that use quantum models with fused features
