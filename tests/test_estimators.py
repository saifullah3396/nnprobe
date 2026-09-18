from typing import cast

import numpy as np
import pytest

from nnprobe import ProbeConfig, ProbeKind
from nnprobe._estimators import build_estimator

X = np.array([[2.0, 0.0], [3.0, 0.0], [-2.0, 0.0], [-3.0, 0.0]])
y = np.array(["positive", "positive", "negative", "negative"])


@pytest.mark.parametrize("kind", ["logistic", "linear_svm", "mean_difference"])
def test_probe_kinds_fit_and_predict(kind: str) -> None:
    estimator = build_estimator(
        config=ProbeConfig(
            kind=cast(ProbeKind, kind), add_scaling=False, max_iter=1_000
        )
    )
    estimator.fit(activations=X, targets=y)

    np.testing.assert_array_equal(estimator.predict(activations=X), y)
    assert estimator.scores(activations=X).shape[0] == len(X)


def test_logistic_probe_has_probabilities() -> None:
    estimator = build_estimator(
        config=ProbeConfig(kind="logistic", add_scaling=False, max_iter=1_000)
    )
    estimator.fit(activations=X, targets=y)

    probabilities = estimator.probabilities(activations=X)
    assert probabilities is not None
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_linear_svm_has_no_probabilities() -> None:
    estimator = build_estimator(
        config=ProbeConfig(kind="linear_svm", add_scaling=False, max_iter=1_000)
    )
    estimator.fit(activations=X, targets=y)

    assert estimator.probabilities(activations=X) is None


def test_mean_difference_requires_two_classes() -> None:
    estimator = build_estimator(config=ProbeConfig(kind="mean_difference"))

    with pytest.raises(ValueError, match="at least two classes"):
        estimator.fit(activations=X[:1], targets=np.array(["only"]))
