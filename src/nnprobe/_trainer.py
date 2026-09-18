from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, final

import numpy as np
from nnact import ActivationDataset

from nnprobe._config import ProbeConfig
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
        self.estimator_: object | None = None
        self.classes_: np.ndarray | None = None

    def train(
        self,
        *,
        dataset: ActivationDataset,
        layer_name: str,
        filter_fn: FilterFn | None = None,
        pool_fn: PoolFn | None = None,
    ) -> TrainResult:
        x, y, sample_of_row = self._select(
            dataset, layer_name, filter_fn=filter_fn, pool_fn=pool_fn
        )
        train_mask, test_mask = self._split(sample_of_row)
        return self._fit(x[train_mask], y[train_mask], x[test_mask], y[test_mask])

    def evaluate(
        self,
        *,
        dataset: ActivationDataset,
        layer_name: str,
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
            dataset, layer_name, filter_fn=filter_fn, pool_fn=pool_fn
        )
        unknown_labels = set(np.unique(y).tolist()) - set(self.classes_.tolist())
        if unknown_labels:
            raise ValueError(
                f"dataset has label(s) {unknown_labels} the probe was never "
                f"trained on; known classes are {self.classes_.tolist()}."
            )
        y_idx = np.searchsorted(self.classes_, y)
        predictions, probabilities = self._predict(x)
        return EvalResult(
            predictions=predictions,
            probabilities=probabilities,
            targets=y_idx,
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
        filter_fn: FilterFn | None,
        pool_fn: PoolFn | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        metadata = dataset.metadata
        if metadata is None or "labels" not in metadata:
            raise ValueError("dataset metadata must contain a 'labels' entry")
        labels = np.asarray(metadata["labels"])

        is_token_level = hasattr(dataset, "sample_of_token")

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
        import cuml
        import cuml.pipeline
        import cupy

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

        classes = np.unique(y_train)
        y_train_idx = np.searchsorted(classes, y_train)
        y_test_idx = np.searchsorted(classes, y_test)

        logger.info(
            "DEBUG fit features shape=%s dtype=%s; train_labels shape=%s dtype=%s "
            "classes=%s; test_labels shape=%s dtype=%s",
            x_train.shape,
            x_train.dtype,
            y_train_idx.shape,
            y_train_idx.dtype,
            classes.tolist(),
            y_test_idx.shape,
            y_test_idx.dtype,
        )

        steps = []
        if self._config.add_scaling:
            steps.append(("scaler", cuml.preprocessing.StandardScaler()))
        steps.append(
            (
                "clf",
                cuml.linear_model.LogisticRegression(
                    C=self._config.C,
                    max_iter=self._config.max_iter,
                    linesearch_max_iter=self._config.linesearch_max_iter,
                ),
            )
        )
        estimator = cuml.pipeline.Pipeline(steps) if len(steps) > 1 else steps[0][1]

        cupy_x_train = cupy.asarray(x_train)
        cupy_y_train = cupy.asarray(y_train_idx)

        estimator.fit(cupy_x_train, cupy_y_train)
        self.estimator_ = estimator
        self.classes_ = classes

        predictions, probabilities = self._predict(x_test)
        return TrainResult(
            predictions=predictions,
            probabilities=probabilities,
            targets=y_test_idx,
            estimator=estimator,
            classes_=classes,
        )

    def _predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import cupy

        estimator = self.estimator_
        cupy_x = cupy.asarray(x)

        predictions = cupy.asnumpy(estimator.predict(cupy_x))
        probabilities = cupy.asnumpy(estimator.predict_proba(cupy_x))
        return predictions, probabilities
