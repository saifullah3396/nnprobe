from __future__ import annotations

from typing import Protocol

import numpy as np

from nnprobe._config import ProbeConfig


class ProbeEstimator(Protocol):
    classes_: np.ndarray

    def fit(self, *, activations: np.ndarray, targets: np.ndarray) -> None: ...

    def predict(self, *, activations: np.ndarray) -> np.ndarray: ...

    def scores(self, *, activations: np.ndarray) -> np.ndarray: ...

    def probabilities(self, *, activations: np.ndarray) -> np.ndarray | None: ...


class _SklearnEstimator:
    def __init__(self, *, config: ProbeConfig) -> None:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if config.kind == "logistic":
            from sklearn.linear_model import LogisticRegression

            estimator = LogisticRegression(C=config.C, max_iter=config.max_iter)
        else:
            from sklearn.svm import LinearSVC

            estimator = LinearSVC(C=config.C, max_iter=config.max_iter)
        self._estimator = (
            make_pipeline(StandardScaler(), estimator)
            if config.add_scaling
            else estimator
        )

    def fit(self, *, activations: np.ndarray, targets: np.ndarray) -> None:
        self._estimator.fit(activations, targets)
        self.classes_ = np.asarray(self._estimator.classes_)

    def predict(self, *, activations: np.ndarray) -> np.ndarray:
        return np.asarray(self._estimator.predict(activations))

    def scores(self, *, activations: np.ndarray) -> np.ndarray:
        return np.asarray(self._estimator.decision_function(activations))

    def probabilities(self, *, activations: np.ndarray) -> np.ndarray | None:
        if hasattr(self._estimator, "predict_proba"):
            return np.asarray(self._estimator.predict_proba(activations))
        return None


class MeanDifferenceEstimator:
    def __init__(self, *, normalize: bool) -> None:
        self._normalize = normalize

    def fit(self, *, activations: np.ndarray, targets: np.ndarray) -> None:
        self.classes_ = np.unique(targets)
        if len(self.classes_) < 2:
            raise ValueError("mean-difference probe requires at least two classes")
        directions = []
        for target in self.classes_:
            direction = activations[targets == target].mean(axis=0) - activations[
                targets != target
            ].mean(axis=0)
            if self._normalize:
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
            directions.append(direction)
        self.directions_ = np.stack(directions)

    def scores(self, *, activations: np.ndarray) -> np.ndarray:
        return activations @ self.directions_.T

    def predict(self, *, activations: np.ndarray) -> np.ndarray:
        return self.classes_[self.scores(activations=activations).argmax(axis=1)]

    def probabilities(self, *, activations: np.ndarray) -> np.ndarray | None:
        return None


def build_estimator(*, config: ProbeConfig) -> ProbeEstimator:
    if config.kind == "mean_difference":
        return MeanDifferenceEstimator(normalize=config.normalize_direction)
    return _SklearnEstimator(config=config)
