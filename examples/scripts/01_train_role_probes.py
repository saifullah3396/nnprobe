import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import fire
import numpy as np
import pandas as pd
import torch
from nnact import (
    ActivationDataset,
    ActivationPipeline,
    SequenceModelInput,
    TokenActivationSample,
)
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from nnprobe import ProbeConfig, ProbePipeline, ProbeTrainer, TargetFn
from nnprobe._trainer import FilterFn


class RoleConversationSamples(Dataset[TokenActivationSample]):
    def __init__(
        self,
        path: Path,
        n: int | None = None,
        seed: int = 0,
        no_role_label: str = "none",
    ) -> None:
        with path.open() as f:
            records = [json.loads(line) for line in f]

        if n is not None and n < len(records):
            records = random.Random(seed).sample(records, n)

        self._records = records
        self._no_role_label = no_role_label

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> TokenActivationSample:
        record = self._records[idx]
        target_role = record["metadata"]["target_role"]
        if target_role == "thinking":
            target_role = "cot"
        labels = [
            role if role == target_role else self._no_role_label
            for role in record["token_roles"]
        ]
        turn_positions = [
            position if position is not None else -1
            for position in record["token_idx_in_turn"]
        ]
        return TokenActivationSample(
            model_input=SequenceModelInput(
                input_ids=torch.tensor(record["token_ids"], dtype=torch.long),
                attention_mask=torch.tensor(record["attention_mask"], dtype=torch.long),
            ),
            metadata={"role": labels, "turn_position": turn_positions},
        )


def make_role_targets() -> TargetFn:
    def role_targets(*, dataset) -> np.ndarray:
        metadata = dataset.metadata
        if metadata is None or "role" not in metadata:
            raise ValueError("dataset metadata must contain 'role'")
        return metadata["role"]

    return role_targets


def make_drop_outside_role_space(
    role_space: Sequence[str], skip_first_n: int
) -> FilterFn:
    def drop_outside_role_space(*, dataset) -> np.ndarray:
        metadata = dataset.metadata
        assert metadata is not None
        return np.isin(metadata["role"], role_space) & (
            metadata["turn_position"] >= skip_first_n
        )

    return drop_outside_role_space


def main(
    model: str = "Qwen/Qwen3-1.7B",
    fake_dataset_path: str | None = None,
    roles: tuple[str, ...] = ("user", "assistant", "system", "tool", "cot"),
    no_role_label: str = "none",
    layer: int = 16,
    num_samples: int | None = None,
    sample_seed: int = 0,
    probe_cache_dir: str = "runs/01_train_role_probes/probes",
    skip_first_n: int = 32,
    role_combinations: tuple[tuple[str, ...], ...] = (
        ("user", "assistant"),
        ("user", "assistant", "tool"),
        ("user", "cot", "assistant"),
    ),
) -> None:
    dataset_path = (
        Path(fake_dataset_path)
        if fake_dataset_path is not None
        else Path(__file__).with_name("fake_dataset.jsonl")
    )
    cache_dir = Path(probe_cache_dir)
    # The reference probes `all_pre_mlp_hidden_states` -- the layernormed
    # hidden state going into the MLP, not the decoder block's full output.
    # `post_attention_layernorm` is that exact tensor's producing submodule.
    layer_name = f"model.layers.{layer}.post_attention_layernorm"
    role_space_cache_paths = {
        ",".join(role_space): cache_dir / ("-".join(role_space) + ".npz")
        for role_space in role_combinations
    }
    all_probes_cached = all(path.exists() for path in role_space_cache_paths.values())

    activations = None
    if not all_probes_cached:
        tokenizer = AutoTokenizer.from_pretrained(model)
        model_instance = AutoModelForCausalLM.from_pretrained(model, dtype="auto")

        dataset = RoleConversationSamples(
            dataset_path,
            n=num_samples,
            seed=sample_seed,
            no_role_label=no_role_label,
        )
        print(f"{len(dataset)} conversations | roles: {list(roles)}")
        activation_pipeline = ActivationPipeline(
            model=model_instance,
            layer_names=[layer_name],
            output_type="token",
            tokenizer=tokenizer,
            cache_outputs=True,
            cache_dir=cache_dir.parent,
        )
        activations = activation_pipeline.run(dataset=dataset, batch_size=1).dataset
        print(activations.summary())
    else:
        print("All role-space probes already cached; skipping activation extraction.")

    probe_pipeline = ProbePipeline(
        trainer=ProbeTrainer(config=ProbeConfig(C=1.0e-1, add_scaling=False))
    )
    target_fn = make_role_targets()
    rows: list[dict[str, float | str]] = []
    for role_space in role_combinations:
        role_space_key = ",".join(role[0] for role in role_space)
        cache_path = role_space_cache_paths[",".join(role_space)]
        filter_fn = None
        if activations is not None:
            filter_fn = make_drop_outside_role_space(role_space, skip_first_n)
            eligible = filter_fn(
                dataset=activations,
            )
            print(
                f"DEBUG role_space={role_space} eligible_tokens={int(eligible.sum())} "
                f"skip_first_n={skip_first_n}"
            )
        result = probe_pipeline.train(
            dataset=cast(ActivationDataset, activations),
            layer_name=layer_name,
            target_fn=target_fn,
            cache_path=cache_path,
            filter_fn=filter_fn,
        )
        metrics = result.metrics
        confusion_table = pd.DataFrame(
            metrics.confusion_matrix,
            index=pd.Index(result.classes_, name="true"),
            columns=pd.Index(result.classes_, name="predicted"),
        )
        print(f"\nconfusion matrix [{role_space_key}]:\n{confusion_table}\n")
        rows.append(
            {
                "layer": layer,
                "role_space": role_space_key,
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
            }
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    results_path = cache_dir / "results.json"
    json_rows = [{key: value for key, value in row.items()} for row in rows]
    results_path.write_text(json.dumps(json_rows, indent=2))

    print(pd.DataFrame(rows))


if __name__ == "__main__":
    fire.Fire(main)
