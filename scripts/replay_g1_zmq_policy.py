#!/usr/bin/env python3
"""Replay recorded G1 SONIC actions through the GR00T ZMQ policy protocol."""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import polars as pl


ACTION_COLUMNS = {
    "motion_token": ("action.motion_token", 64),
    "left_hand_joints": ("teleop.left_hand_joints", 7),
    "right_hand_joints": ("teleop.right_hand_joints", 7),
}


@dataclasses.dataclass(frozen=True)
class ReplayEpisode:
    """Validated action arrays and timing metadata for one recorded episode."""

    dataset_dir: pathlib.Path
    episode_id: int
    fps: float
    actions: dict[str, np.ndarray]

    @property
    def length(self) -> int:
        return self.actions["motion_token"].shape[0]

    @classmethod
    def load(cls, dataset_dir: pathlib.Path, episode_id: int) -> ReplayEpisode:
        if episode_id < 0:
            raise ValueError(f"episode_id must be non-negative, got {episode_id}")

        dataset_dir = pathlib.Path(dataset_dir)
        metadata_path = dataset_dir / "meta" / "info.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing dataset metadata: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        try:
            fps = float(metadata["fps"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Dataset metadata must contain a numeric fps: {metadata_path}") from exc
        if fps <= 0:
            raise ValueError(f"Dataset fps must be positive, got {fps}")

        parquet_path = (
            dataset_dir
            / "data"
            / f"chunk-{episode_id // 1000:03d}"
            / f"episode_{episode_id:06d}.parquet"
        )
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Missing episode parquet: {parquet_path}")

        required_columns = [column for column, _ in ACTION_COLUMNS.values()]
        schema = pl.read_parquet_schema(parquet_path)
        missing_columns = [column for column in required_columns if column not in schema]
        if missing_columns:
            raise ValueError(f"Missing required action columns: {missing_columns}")

        frame = pl.read_parquet(parquet_path, columns=required_columns)
        if frame.height == 0:
            raise ValueError(f"Episode {episode_id} must contain at least one frame")

        actions: dict[str, np.ndarray] = {}
        for output_key, (column, width) in ACTION_COLUMNS.items():
            action = np.asarray(frame[column].to_list(), dtype=np.float32)
            expected_shape = (frame.height, width)
            if action.shape != expected_shape:
                raise ValueError(f"{column} must have shape [T, {width}], got {action.shape}")
            actions[output_key] = np.ascontiguousarray(action)

        return cls(dataset_dir=dataset_dir, episode_id=episode_id, fps=fps, actions=actions)


def _slice_with_final_padding(array: np.ndarray, start_frame: int, horizon: int) -> np.ndarray:
    clamped_start = min(start_frame, array.shape[0] - 1)
    chunk = array[clamped_start : clamped_start + horizon]
    missing = horizon - chunk.shape[0]
    if missing == 0:
        return chunk
    return np.concatenate((chunk, np.repeat(array[-1:], missing, axis=0)), axis=0)


class ReplayTimeline:
    """Select overlapping recorded chunks using request time rather than call count."""

    def __init__(self, episode: ReplayEpisode, *, horizon: int) -> None:
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        self._episode = episode
        self._horizon = horizon
        self._anchor_time: float | None = None

    def reset(self) -> None:
        self._anchor_time = None

    def get_action(self, request_time: float) -> tuple[dict[str, np.ndarray], int]:
        if self._anchor_time is None:
            self._anchor_time = request_time
        elapsed = max(0.0, request_time - self._anchor_time)
        start_frame = int(np.round(elapsed * self._episode.fps))
        action = {
            key: _slice_with_final_padding(value, start_frame, self._horizon)
            for key, value in self._episode.actions.items()
        }
        return action, start_frame
