#!/usr/bin/env python3
"""
classical_detector.py — the classical anomaly-detection floor.

MahalanobisDetector: models the normal hours as a single multivariate Gaussian
(robust LedoitWolf covariance to stay stable when features >> samples) and scores
each hour by its Mahalanobis distance from that normal manifold. This is the
interpretable, no-training baseline the DL model must beat, and the natural home
for the LEAN feature set (covariance estimation degrades badly in ~200-D).

Interface shared with the DL detector:
    d = MahalanobisDetector().fit(X_normal)
    scores = d.score(X)          # higher = more anomalous (per row)
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


class MahalanobisDetector:
    name = "Classical"

    def __init__(self):
        self.cov_ = None

    def fit(self, X):
        self.cov_ = LedoitWolf().fit(np.asarray(X, dtype=float))
        return self

    def score(self, X):
        d2 = self.cov_.mahalanobis(np.asarray(X, dtype=float))
        return np.sqrt(np.clip(d2, 0.0, None))
