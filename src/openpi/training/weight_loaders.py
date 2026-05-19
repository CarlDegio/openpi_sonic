import dataclasses
import logging
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class ShapeAwareCheckpointWeightLoader(WeightLoader):
    """Loads checkpoint weights while reinitializing explicitly allowed mismatched parameters.

    This is useful when adapting a checkpoint to a new action dimensionality. Backbone
    parameters are still required to match exactly; only parameters matching
    `reinit_mismatched_regexes` may keep their current initialization.
    """

    params_path: str
    reinit_mismatched_regexes: Sequence[str] = ()
    missing_regex: str = ".*lora.*"

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params_shape_aware(
            loaded_params,
            params,
            missing_regex=self.missing_regex,
            reinit_mismatched_regexes=self.reinit_mismatched_regexes,
        )


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _merge_params_shape_aware(
    loaded_params: at.Params,
    params: at.Params,
    *,
    missing_regex: str,
    reinit_mismatched_regexes: Sequence[str],
) -> at.Params:
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    missing_pattern = re.compile(missing_regex)
    reinit_patterns = [re.compile(pattern) for pattern in reinit_mismatched_regexes]

    def should_reinit(key: str) -> bool:
        return any(pattern.fullmatch(key) for pattern in reinit_patterns)

    result = {}
    for key, value in flat_loaded.items():
        if key not in flat_ref:
            continue
        ref_value = flat_ref[key]
        if value.shape == ref_value.shape:
            result[key] = value.astype(ref_value.dtype) if value.dtype != ref_value.dtype else value
            continue
        if should_reinit(key):
            logger.info(
                "Keeping initialized parameter for shape-mismatched checkpoint key %s: checkpoint=%s model=%s",
                key,
                value.shape,
                ref_value.shape,
            )
            continue
        raise ValueError(
            f"Shape mismatch at {key}: checkpoint has {value.shape}, model expects {ref_value.shape}. "
            "Add an explicit reinit pattern if this parameter should be reinitialized."
        )

    for key, value in flat_ref.items():
        if key in result:
            continue
        if missing_pattern.fullmatch(key) or should_reinit(key):
            result[key] = value

    return flax.traverse_util.unflatten_dict(result, sep="/")
