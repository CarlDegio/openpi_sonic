from __future__ import annotations

import json
import pathlib

import numpy as np
import polars as pl
import pytest

from scripts.replay_g1_zmq_policy import ReplayEpisode
from scripts.replay_g1_zmq_policy import ReplayTimeline


ACTION_COLUMNS = {
    "motion_token": ("action.motion_token", 64),
    "left_hand_joints": ("teleop.left_hand_joints", 7),
    "right_hand_joints": ("teleop.right_hand_joints", 7),
}


def _write_episode(
    root: pathlib.Path,
    *,
    episode_id: int = 0,
    length: int = 130,
    fps: float = 50.0,
    widths: dict[str, int] | None = None,
    omitted_columns: set[str] | None = None,
) -> pathlib.Path:
    widths = widths or {}
    omitted_columns = omitted_columns or set()

    (root / "meta").mkdir(parents=True)
    parquet_dir = root / "data" / f"chunk-{episode_id // 1000:03d}"
    parquet_dir.mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"fps": fps}), encoding="utf-8")

    frames = np.arange(length, dtype=np.float32)
    columns = {
        column: [np.full(widths.get(name, width), frame, dtype=np.float32).tolist() for frame in frames]
        for name, (column, width) in ACTION_COLUMNS.items()
        if column not in omitted_columns
    }
    pl.DataFrame(columns).write_parquet(parquet_dir / f"episode_{episode_id:06d}.parquet")
    return root


def test_load_episode_reads_selected_chunk_and_float32_actions(tmp_path: pathlib.Path):
    dataset_dir = _write_episode(tmp_path / "dataset", episode_id=1001, length=3)

    episode = ReplayEpisode.load(dataset_dir, episode_id=1001)

    assert episode.dataset_dir == dataset_dir
    assert episode.episode_id == 1001
    assert episode.fps == 50.0
    assert episode.length == 3
    assert episode.actions["motion_token"].shape == (3, 64)
    assert episode.actions["left_hand_joints"].shape == (3, 7)
    assert episode.actions["right_hand_joints"].shape == (3, 7)
    assert all(action.dtype == np.float32 for action in episode.actions.values())
    np.testing.assert_array_equal(episode.actions["motion_token"][:, 0], np.array([0, 1, 2]))


def test_load_episode_rejects_negative_episode_id(tmp_path: pathlib.Path):
    with pytest.raises(ValueError, match="episode_id must be non-negative"):
        ReplayEpisode.load(tmp_path, episode_id=-1)


def test_load_episode_rejects_missing_metadata(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError, match="metadata"):
        ReplayEpisode.load(tmp_path, episode_id=0)


def test_load_episode_rejects_missing_parquet(tmp_path: pathlib.Path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps({"fps": 50}), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="episode parquet"):
        ReplayEpisode.load(tmp_path, episode_id=0)


@pytest.mark.parametrize("fps", [0.0, -50.0])
def test_load_episode_rejects_non_positive_fps(tmp_path: pathlib.Path, fps: float):
    dataset_dir = _write_episode(tmp_path / "dataset", fps=fps)

    with pytest.raises(ValueError, match="fps must be positive"):
        ReplayEpisode.load(dataset_dir, episode_id=0)


def test_load_episode_rejects_missing_action_column(tmp_path: pathlib.Path):
    dataset_dir = _write_episode(
        tmp_path / "dataset",
        omitted_columns={"teleop.right_hand_joints"},
    )

    with pytest.raises(ValueError, match="Missing required action columns.*teleop.right_hand_joints"):
        ReplayEpisode.load(dataset_dir, episode_id=0)


def test_load_episode_rejects_empty_episode(tmp_path: pathlib.Path):
    dataset_dir = _write_episode(tmp_path / "dataset", length=0)

    with pytest.raises(ValueError, match="must contain at least one frame"):
        ReplayEpisode.load(dataset_dir, episode_id=0)


def test_load_episode_rejects_wrong_action_width(tmp_path: pathlib.Path):
    dataset_dir = _write_episode(tmp_path / "dataset", widths={"motion_token": 63})

    with pytest.raises(ValueError, match=r"action.motion_token must have shape \[T, 64\]"):
        ReplayEpisode.load(dataset_dir, episode_id=0)


def test_timeline_uses_wall_clock_and_preserves_overlapping_frames(tmp_path: pathlib.Path):
    episode = ReplayEpisode.load(_write_episode(tmp_path / "dataset"), episode_id=0)
    timeline = ReplayTimeline(episode, horizon=70)

    first, first_start = timeline.get_action(100.0)
    second, second_start = timeline.get_action(101.0)

    assert first_start == 0
    assert second_start == 50
    np.testing.assert_array_equal(first["motion_token"][50:], second["motion_token"][:20])
    np.testing.assert_array_equal(second["motion_token"][:, 0], np.arange(50, 120, dtype=np.float32))


def test_timeline_pads_partial_chunk_with_final_frame(tmp_path: pathlib.Path):
    episode = ReplayEpisode.load(_write_episode(tmp_path / "dataset", length=55), episode_id=0)
    timeline = ReplayTimeline(episode, horizon=10)
    timeline.get_action(10.0)

    action, start_frame = timeline.get_action(11.0)

    assert start_frame == 50
    np.testing.assert_array_equal(
        action["motion_token"][:, 0],
        np.array([50, 51, 52, 53, 54, 54, 54, 54, 54, 54], dtype=np.float32),
    )


def test_timeline_returns_only_final_frame_beyond_episode(tmp_path: pathlib.Path):
    episode = ReplayEpisode.load(_write_episode(tmp_path / "dataset", length=3), episode_id=0)
    timeline = ReplayTimeline(episode, horizon=70)
    timeline.get_action(10.0)

    action, start_frame = timeline.get_action(20.0)

    assert start_frame == 500
    assert action["motion_token"].shape == (70, 64)
    np.testing.assert_array_equal(action["motion_token"], np.full((70, 64), 2, dtype=np.float32))
    np.testing.assert_array_equal(action["left_hand_joints"], np.full((70, 7), 2, dtype=np.float32))
    np.testing.assert_array_equal(action["right_hand_joints"], np.full((70, 7), 2, dtype=np.float32))


def test_timeline_reset_reanchors_next_request_at_frame_zero(tmp_path: pathlib.Path):
    episode = ReplayEpisode.load(_write_episode(tmp_path / "dataset"), episode_id=0)
    timeline = ReplayTimeline(episode, horizon=70)
    timeline.get_action(100.0)
    _, before_reset_start = timeline.get_action(101.0)

    timeline.reset()
    action, after_reset_start = timeline.get_action(500.0)

    assert before_reset_start == 50
    assert after_reset_start == 0
    assert action["motion_token"][0, 0] == 0


def test_timeline_rejects_non_positive_horizon(tmp_path: pathlib.Path):
    episode = ReplayEpisode.load(_write_episode(tmp_path / "dataset"), episode_id=0)

    with pytest.raises(ValueError, match="horizon must be positive"):
        ReplayTimeline(episode, horizon=0)
