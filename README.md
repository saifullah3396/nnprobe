# nnprobe

`nnprobe` trains linear probes over activations produced by
[`nnact`](../nnact). The packages have deliberately separate responsibilities:

- `nnact` runs models, captures activations, and provides activation datasets.
- `nnprobe` selects rows, trains probes, evaluates predictions, and caches
  fitted results.

## Basic usage

```python
from nnprobe import ProbeConfig, ProbePipeline, ProbeTrainer

trainer = ProbeTrainer(config=ProbeConfig(C=0.1))
pipeline = ProbePipeline(trainer=trainer)
result = pipeline.train(
    dataset=activation_dataset,
    layer_name="model.layers.16.post_attention_layernorm",
)
```

Labels must be attached to the `nnact` dataset as row-aligned metadata under
the `"labels"` key. A filter receives the complete dataset and returns one
boolean value for every activation row:

```python
def keep_role_tokens(*, dataset):
    metadata = dataset.metadata
    assert metadata is not None
    return metadata["labels"] != "none"
```

The runnable role-probe example is
`examples/scripts/01_train_role_probes.py`. It uses Fire, so configuration is
provided as command-line arguments rather than module-level constants.
