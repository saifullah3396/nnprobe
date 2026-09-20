import numpy as np
from nnact import ActivationDataset

from nnprobe import ProbeConfig, ProbeTrainer


class FakeSequenceDataset(ActivationDataset):
    def __init__(self) -> None:
        self._activations = {"layer": np.arange(12, dtype=np.float32).reshape(3, 2, 2)}
        self._metadata = {"role": np.array(["keep", "drop", "keep"])}

    @property
    def layer_names(self) -> list[str]:
        return ["layer"]

    def activations(self, layer_name: str) -> np.ndarray:
        return self._activations[layer_name]

    @property
    def metadata(self) -> dict[str, np.ndarray]:
        return self._metadata

    def __len__(self) -> int:
        return 3


def test_filter_receives_the_complete_dataset() -> None:
    dataset = FakeSequenceDataset()
    seen: list[object] = []

    def keep_rows(*, dataset: ActivationDataset) -> np.ndarray:
        seen.append(dataset)
        metadata = getattr(dataset, "metadata", None)
        assert metadata is not None
        return metadata["role"] == "keep"

    trainer = ProbeTrainer(config=ProbeConfig())

    def target_fn(*, dataset: ActivationDataset) -> np.ndarray:
        metadata = getattr(dataset, "metadata", None)
        assert metadata is not None
        return metadata["role"]

    activations, labels, samples, _selected_indices = trainer._select(
        dataset=dataset,
        layer_name="layer",
        target_fn=target_fn,
        filter_fn=keep_rows,
        pool_fn=None,
    )

    assert seen == [dataset]
    np.testing.assert_array_equal(labels, ["keep", "keep"])
    np.testing.assert_array_equal(samples, [0, 2])
    assert activations.shape == (2, 2)
