from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, final

import numpy as np
from nnact import ActivationDataset

from nnprobe._config import ProbeConfig
from nnprobe._estimators import ProbeEstimator, build_estimator
from nnprobe._result import EvalResult, TrainResult

logger = logging.getLogger(__name__)

type Metadata = dict[str, np.ndarray]


class FilterFn(Protocol):
    """Select rows from one :class:`nnact.ActivationDataset`.

    The dataset is passed intact so filters can use its activations, metadata,
    token ids, and token-to-sample mapping without a second parallel API.
    """

    def __call__(
        self,
        *,
        dataset: ActivationDataset,
    ) -> np.ndarray:
        """Return a boolean mask over this dataset's activation rows."""
        ...


class TargetFn(Protocol):
    """Extract one training target for every activation row."""

    def __call__(self, *, dataset: ActivationDataset) -> np.ndarray: ...


class PoolFn(Protocol):
    """Reduce sequence activations while preserving the sample row count."""

    def __call__(self, *, activations: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Reduce ``(num_rows, seq_len, *feature)`` to ``(num_rows, *feature)``."""
        ...


def _default_pool(*, activations: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return activations[:, 0, ...]


@final
class ProbeTrainer:
    """Fits and evaluates one linear probe on an :class:`ActivationDataset`.

    Example:
        >>> trainer = ProbeTrainer(config=ProbeConfig(C=0.5))
        >>> result = trainer.train(  # doctest: +SKIP
        ...     dataset=dataset, layer_name="model.layers.12"
        ... )
    """

    def __init__(self, *, config: ProbeConfig) -> None:
        self._config = config
        self.estimator_: ProbeEstimator | None = None
        self.classes_: np.ndarray | None = None

    def train(
        self,
        *,
        dataset: ActivationDataset,
        layer_name: str,
        target_fn: TargetFn,
        filter_fn: FilterFn | None = None,
        pool_fn: PoolFn | None = None,
    ) -> TrainResult:
        x, y, sample_of_row = self._select(
            dataset,
            layer_name,
            target_fn=target_fn,
            filter_fn=filter_fn,
            pool_fn=pool_fn,
        )
        train_mask, test_mask = self._split(sample_of_row)
        return self._fit(x[train_mask], y[train_mask], x[test_mask], y[test_mask])

    def evaluate(
        self,
        *,
        dataset: ActivationDataset,
        layer_name: str,
        target_fn: TargetFn,
        filter_fn: FilterFn | None = None,
        pool_fn: PoolFn | None = None,
    ) -> EvalResult:
        """Score an already-fitted probe against fresh examples.

        Unlike :meth:`train`, every selected row is scored -- there is no
        train/test split, since the probe was already fitted elsewhere
        (typically by an earlier :meth:`train` call on this same trainer).
        """
        if self.estimator_ is None or self.classes_ is None:
            raise RuntimeError("no fitted estimator; call train() before evaluate()")

        x, y, _sample_of_row = self._select(
            dataset,
            layer_name,
            target_fn=target_fn,
            filter_fn=filter_fn,
            pool_fn=pool_fn,
        )
        unknown_labels = set(np.unique(y).tolist()) - set(self.classes_.tolist())
        if unknown_labels:
            raise ValueError(
                f"dataset has label(s) {unknown_labels} the probe was never "
                f"trained on; known classes are {self.classes_.tolist()}."
            )
        predictions, probabilities = self._predict(x)
        return EvalResult(
            predictions=predictions,
            probabilities=probabilities,
            scores=self.estimator_.scores(activations=x),
            targets=y,
            classes_=self.classes_,
        )

    def load(self, *, path: str | Path) -> TrainResult:
        """Restore the fitted estimator from a :class:`TrainResult` cached by
        :meth:`~nnprobe._probing._pipeline.ProbePipeline.train`, so
        :meth:`evaluate` can run against it without calling :meth:`train`
        again first.
        """
        result = TrainResult.load(path=path)
        self.estimator_ = result.estimator
        self.classes_ = result.classes_
        return result

    def _select(
        self,
        dataset: ActivationDataset,
        layer_name: str,
        *,
        target_fn: TargetFn,
        filter_fn: FilterFn | None,
        pool_fn: PoolFn | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        labels = np.asarray(target_fn(dataset=dataset))

        if labels.ndim != 1:
            raise ValueError(
                f"target_fn must return a one-dimensional array, got {labels.shape}"
            )

        is_token_level = hasattr(dataset, "sample_of_token")
        expected_rows = (
            dataset.activations[layer_name].shape[0] if is_token_level else len(dataset)
        )
        if labels.shape != (expected_rows,):
            raise ValueError(
                f"target_fn returned {labels.shape}; expected ({expected_rows},)"
            )

        if is_token_level:
            if pool_fn is not None:
                raise ValueError("pool_fn cannot be used with token-level datasets")
            x = dataset.activations[layer_name]
            y = labels
            sample_of_row = dataset.sample_of_token  # type: ignore[attr-defined]
            mask = self._filter_mask(filter_fn, dataset, sample_of_row)
            x, y, sample_of_row = x[mask], y[mask], sample_of_row[mask]
        else:
            raw = dataset.activations[layer_name]
            y = labels
            sample_of_row = np.arange(len(dataset))

            mask = self._filter_mask(filter_fn, dataset, sample_of_row)
            raw, y, sample_of_row = raw[mask], y[mask], sample_of_row[mask]

            pool = pool_fn or _default_pool
            x = pool(activations=raw, labels=y)
            if x.shape[0] != raw.shape[0]:
                raise ValueError(
                    f"pooled activations must preserve the row count: "
                    f"expected {raw.shape[0]}, got {x.shape[0]}"
                )

        return x, y, sample_of_row

    def _filter_mask(
        self,
        filter_fn: FilterFn | None,
        dataset: ActivationDataset,
        sample_of_row: np.ndarray,
    ) -> np.ndarray:
        if filter_fn is None:
            return np.ones(sample_of_row.shape[0], dtype=bool)
        mask = filter_fn(dataset=dataset)
        if mask.shape != sample_of_row.shape:
            raise ValueError(
                f"filter_fn mask must have shape {sample_of_row.shape}, got {mask.shape}"
            )
        return mask

    def _split(self, sample_of_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.model_selection import train_test_split

        unique_samples = np.unique(sample_of_row)
        train_samples, test_samples = train_test_split(
            unique_samples,
            test_size=self._config.test_size,
            random_state=self._config.seed,
        )
        train_mask = np.isin(sample_of_row, train_samples)
        test_mask = np.isin(sample_of_row, test_samples)
        return train_mask, test_mask

    def _fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> TrainResult:
        logger.info(
            "Fitting probe: %d train rows, %d test rows", len(y_train), len(y_test)
        )
        missing_train_labels = set(np.unique(y_test).tolist()) - set(
            np.unique(y_train).tolist()
        )
        if missing_train_labels:
            raise ValueError(
                f"Training split is missing label(s) {missing_train_labels} "
                "present in the test split -- check that filter_fn/labels "
                "actually include tokens for every role in this role space."
            )

        estimator = build_estimator(config=self._config)
        estimator.fit(activations=x_train, targets=y_train)
        self.estimator_ = estimator
        self.classes_ = estimator.classes_

        predictions, probabilities = self._predict(x_test)
        return TrainResult(
            predictions=predictions,
            probabilities=probabilities,
            scores=estimator.scores(activations=x_test),
            targets=y_test,
            estimator=estimator,
            classes_=estimator.classes_,
        )

    def _predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        estimator = self.estimator_
        assert estimator is not None
        predictions = estimator.predict(activations=x)
        probabilities = estimator.probabilities(activations=x)
        return predictions, probabilities
