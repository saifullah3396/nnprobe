from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class ProbeConfig:
    """Hyperparameters used to construct one linear classification probe.

    Attributes:
        C: Inverse regularization strength.
        max_iter: Maximum classifier iterations.
        add_scaling: Whether to standardize features before classification.
        linesearch_max_iter: Maximum line-search iterations.
        test_size: Fraction of samples held out for evaluation.
        seed: Random seed used for the sample split.
    """

    C: float = 1.0
    max_iter: int = 5_000
    add_scaling: bool = True
    linesearch_max_iter: int = 100
    test_size: float = 0.1
    seed: int = 123
