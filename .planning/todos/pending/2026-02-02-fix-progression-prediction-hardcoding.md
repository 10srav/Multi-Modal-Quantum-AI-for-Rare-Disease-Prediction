---
created: 2026-02-02T23:02
title: Fix hardcoded progression prediction
area: api
files:
  - dashboard.py
  - api.py
  - src/models.py
---

## Problem

The progression prediction (Slow / Moderate / Rapid) is hardcoded based on the risk score — it's not the output of any real model, quantum or classical.

This defeats the purpose of the multi-task classification architecture where models output both `risk_logits` and `progression_logits`. The actual progression classifier head is never used.

## Solution

1. Use actual `progression_logits` output from trained models
2. Remove hardcoded progression logic based on risk score
3. Display true model confidence for each progression class
4. Ensure both classical and quantum models output real progression predictions
