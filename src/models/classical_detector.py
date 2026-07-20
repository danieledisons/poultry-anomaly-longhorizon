#!/usr/bin/env python3
"""Mahalanobis-distance detector used as the classical baseline.
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
