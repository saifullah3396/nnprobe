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
    target_fn=role_targets,
)
```

Targets are extracted explicitly from row-aligned dataset metadata with a
`TargetFn`. A filter receives the complete dataset and returns one boolean
value for every activation row:

```python
def role_targets(*, dataset):
    metadata = dataset.metadata
    if metadata is None:
        raise ValueError("missing metadata")
    return metadata["role"]


def keep_role_tokens(*, dataset):
    metadata = dataset.metadata
    if metadata is None:
        raise ValueError("missing metadata")
    return metadata["role"] != "none"
```

The runnable role-probe example is
`examples/scripts/01_train_role_probes.py`. It uses Fire, so configuration is
provided as command-line arguments rather than module-level constants.
