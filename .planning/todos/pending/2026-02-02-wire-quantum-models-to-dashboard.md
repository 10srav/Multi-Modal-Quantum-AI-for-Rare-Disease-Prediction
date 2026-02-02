---
created: 2026-02-02T23:02
title: Wire quantum models (QSVM/QNN) to dashboard
area: ui
files:
  - src/qml/qsvm.py
  - src/qml/qnn.py
  - dashboard.py
  - api.py
---

## Problem

Trained quantum models (QSVM and QNN) exist but aren't wired to the dashboard. Predictions shown in the UI come only from a classical tabular model, not from the QML models.

The quantum inference endpoints may exist in the API but the dashboard never calls them. Users cannot see or compare quantum vs classical predictions.

## Solution

1. Add model selection dropdown to dashboard (Classical SVM, Random Forest, QSVM, QNN)
2. Create/update API endpoints for quantum model inference
3. Display quantum vs classical prediction comparison in dashboard
4. Show confidence scores from both model types
