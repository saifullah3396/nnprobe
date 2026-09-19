from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from nnprobe._config import ProbeConfig

logger = logging.getLogger(__name__)


class ProbeEstimator(Protocol):
    classes_: np.ndarray

    def fit(self, *, activations: np.ndarray, targets: np.ndarray) -> None: ...

    def predict(self, *, activations: np.ndarray) -> np.ndarray: ...

    def scores(self, *, activations: np.ndarray) -> np.ndarray: ...

    def probabilities(self, *, activations: np.ndarray) -> np.ndarray | None: ...


class CuMLEstimator:
    def __init__(self, *, config: ProbeConfig) -> None:
        import cuml
        import cuml.pipeline

        if config.kind == "logistic":
            classifier = cuml.linear_model.LogisticRegression(
                C=config.C,
                max_iter=config.max_iter,
                linesearch_max_iter=config.linesearch_max_iter,
            )
        else:
            classifier = cuml.svm.LinearSVC(C=config.C, max_iter=config.max_iter)

        steps = [("clf", classifier)]
        if config.add_scaling:
            steps.insert(0, ("scaler", cuml.preprocessing.StandardScaler()))
        self._estimator = (
            cuml.pipeline.Pipeline(steps) if len(steps) > 1 else steps[0][1]
        )
        logger.info(
            "Created cuML estimator: classifier=%s scaling=%s pipeline=%s",
            type(classifier).__name__,
            config.add_scaling,
            len(steps) > 1,
        )

    def fit(self, *, activations: np.ndarray, targets: np.ndarray) -> None:
        import cupy

        self.classes_, encoded_targets = np.unique(targets, return_inverse=True)
        logger.info(
            "Encoding %d classes for cuML: classes=%s encoded_dtype=%s",
            len(self.classes_),
            self.classes_.tolist(),
            encoded_targets.dtype,
        )
        self._estimator.fit(cupy.asarray(activations), cupy.asarray(encoded_targets))

    def predict(self, *, activations: np.ndarray) -> np.ndarray:
        import cupy

        indices = cupy.asnumpy(
            self._estimator.predict(cupy.asarray(activations))
        ).astype(np.intp, copy=False)
        return self.classes_[indices]

    def scores(self, *, activations: np.ndarray) -> np.ndarray:
        import cupy

        return cupy.asnumpy(
            self._estimator.decision_function(cupy.asarray(activations))
        )

    def probabilities(self, *, activations: np.ndarray) -> np.ndarray | None:
        import cupy

        predict_proba = getattr(self._estimator, "predict_proba", None)
        if predict_proba is None:
            return None
        return cupy.asnumpy(predict_proba(cupy.asarray(activations)))


# Preserve caches created by the earlier private class name.
_CuMLEstimator = CuMLEstimator


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
    return CuMLEstimator(config=config)
