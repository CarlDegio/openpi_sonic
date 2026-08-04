from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import threading

import numpy as np
import polars as pl
import pytest
import zmq

from scripts.replay_g1_zmq_policy import G1ReplayZmqServer
from scripts.replay_g1_zmq_policy import Gr00tMsgpack
from scripts.replay_g1_zmq_policy import ReplayEpisode
from scripts.replay_g1_zmq_policy import ReplayRequestHandler
from scripts.replay_g1_zmq_policy import ReplayTimeline

ACTION_COLUMNS = {
    "motion_token": ("action.motion_token", 64),
    "left_hand_joints": ("teleop.left_hand_joints", 7),
    "right_hand_joints": ("teleop.right_hand_joints", 7),
}


class _FakeClock:
    def __init__(self, now: float) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


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


def test_handler_applies_delay_after_selecting_request_time_chunk(tmp_path: pathlib.Path):
    episode = ReplayEpisode.load(_write_episode(tmp_path / "dataset"), episode_id=0)
    timeline = ReplayTimeline(episode, horizon=70)
    clock = _FakeClock(100.0)
    handler = ReplayRequestHandler(
        timeline,
        episode=episode,
        inference_delay_s=0.2,
        clock=clock,
        sleep=clock.sleep,
    )

    first_action, first_info = handler.handle({"endpoint": "get_action", "data": {"observation": {}}})
    clock.now = 101.0
    second_action, second_info = handler.handle({"endpoint": "get_action", "data": {"observation": {}}})

    assert clock.sleeps == [0.2, 0.2]
    assert first_info["replay"] == {"episode_id": 0, "start_frame": 0, "start_time_s": 0.0}
    assert second_info["replay"] == {"episode_id": 0, "start_frame": 50, "start_time_s": 1.0}
    assert first_info["server_timing"] == {"convert_ms": 0.0, "infer_ms": pytest.approx(200.0)}
    assert first_info["action_shapes"] == {
        "motion_token": [70, 64],
        "left_hand_joints": [70, 7],
        "right_hand_joints": [70, 7],
    }
    np.testing.assert_array_equal(first_action["motion_token"][50:], second_action["motion_token"][:20])


def test_handler_control_endpoints_are_immediate_and_reset_restarts_replay(tmp_path: pathlib.Path):
    episode = ReplayEpisode.load(_write_episode(tmp_path / "dataset"), episode_id=0)
    timeline = ReplayTimeline(episode, horizon=70)
    clock = _FakeClock(10.0)
    handler = ReplayRequestHandler(
        timeline,
        episode=episode,
        inference_delay_s=0.2,
        clock=clock,
        sleep=clock.sleep,
    )
    handler.handle({"endpoint": "get_action", "data": {"observation": {}}})
    clock.now = 11.0

    assert handler.handle({"endpoint": "ping"})["status"] == "ok"
    metadata = handler.handle({"endpoint": "get_metadata"})
    assert metadata == {
        "dataset_dir": str(episode.dataset_dir),
        "episode_id": 0,
        "fps": 50.0,
        "episode_length": 130,
        "horizon": 70,
        "inference_delay_s": 0.2,
    }
    assert handler.handle({"endpoint": "reset"}) == {"status": "ok"}
    reset_action, reset_info = handler.handle({"endpoint": "get_action", "data": {"observation": {}}})

    assert clock.sleeps == [0.2, 0.2]
    assert reset_info["replay"]["start_frame"] == 0
    assert reset_action["motion_token"][0, 0] == 0


def test_handler_kill_changes_running_state_without_delay(tmp_path: pathlib.Path):
    episode = ReplayEpisode.load(_write_episode(tmp_path / "dataset"), episode_id=0)
    clock = _FakeClock(10.0)
    handler = ReplayRequestHandler(
        ReplayTimeline(episode, horizon=70),
        episode=episode,
        inference_delay_s=0.2,
        clock=clock,
        sleep=clock.sleep,
    )

    response = handler.handle({"endpoint": "kill"})

    assert response["status"] == "ok"
    assert handler.running is False
    assert clock.sleeps == []


def test_handler_rejects_missing_observation_and_unknown_endpoint(tmp_path: pathlib.Path):
    episode = ReplayEpisode.load(_write_episode(tmp_path / "dataset"), episode_id=0)
    clock = _FakeClock(10.0)
    handler = ReplayRequestHandler(
        ReplayTimeline(episode, horizon=70),
        episode=episode,
        inference_delay_s=0.2,
        clock=clock,
        sleep=clock.sleep,
    )

    with pytest.raises(ValueError, match=r"must include data\['observation'\]"):
        handler.handle({"endpoint": "get_action", "data": {}})
    with pytest.raises(ValueError, match="Unknown endpoint: unsupported"):
        handler.handle({"endpoint": "unsupported"})
    assert clock.sleeps == []


def test_handler_rejects_negative_inference_delay(tmp_path: pathlib.Path):
    episode = ReplayEpisode.load(_write_episode(tmp_path / "dataset"), episode_id=0)

    with pytest.raises(ValueError, match="inference_delay_s must be non-negative"):
        ReplayRequestHandler(
            ReplayTimeline(episode, horizon=70),
            episode=episode,
            inference_delay_s=-0.1,
        )


def test_msgpack_round_trip_preserves_replay_action_arrays(tmp_path: pathlib.Path):
    episode = ReplayEpisode.load(_write_episode(tmp_path / "dataset"), episode_id=0)
    clock = _FakeClock(10.0)
    handler = ReplayRequestHandler(
        ReplayTimeline(episode, horizon=70),
        episode=episode,
        inference_delay_s=0.0,
        clock=clock,
        sleep=clock.sleep,
    )
    response = handler.handle({"endpoint": "get_action", "data": {"observation": {}}})

    unpacked = Gr00tMsgpack.unpackb(Gr00tMsgpack.packb(response))

    assert isinstance(unpacked, list)
    assert len(unpacked) == 2
    action, info = unpacked
    assert info["replay"]["start_frame"] == 0
    assert action["motion_token"].shape == (70, 64)
    assert action["left_hand_joints"].shape == (70, 7)
    assert action["right_hand_joints"].shape == (70, 7)
    assert all(value.dtype == np.float32 for value in action.values())


def test_zmq_server_round_trips_policy_endpoints_and_stops_on_kill(tmp_path: pathlib.Path):
    episode = ReplayEpisode.load(_write_episode(tmp_path / "dataset"), episode_id=0)
    handler = ReplayRequestHandler(
        ReplayTimeline(episode, horizon=70),
        episode=episode,
        inference_delay_s=0.0,
    )

    probe_context = zmq.Context()
    probe = probe_context.socket(zmq.REP)
    port = probe.bind_to_random_port("tcp://127.0.0.1")
    probe.close()
    probe_context.term()

    server = G1ReplayZmqServer(handler, host="127.0.0.1", port=port, timeout_ms=100)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()

    client_context = zmq.Context()
    client = client_context.socket(zmq.REQ)
    client.setsockopt(zmq.RCVTIMEO, 2000)
    client.setsockopt(zmq.SNDTIMEO, 2000)
    client.connect(f"tcp://127.0.0.1:{port}")

    def request(payload: dict) -> object:
        client.send(Gr00tMsgpack.packb(payload))
        return Gr00tMsgpack.unpackb(client.recv())

    try:
        assert request({"endpoint": "ping"})["status"] == "ok"
        response = request({"endpoint": "get_action", "data": {"observation": {}}})
        action, info = response
        assert info["replay"]["start_frame"] == 0
        assert action["motion_token"].shape == (70, 64)
        assert action["left_hand_joints"].shape == (70, 7)
        assert action["right_hand_joints"].shape == (70, 7)
        assert request({"endpoint": "reset"}) == {"status": "ok"}
        assert request({"endpoint": "kill"})["status"] == "ok"
        server_thread.join(timeout=2.0)
        assert not server_thread.is_alive()
    finally:
        client.close()
        client_context.term()
        if server_thread.is_alive():
            handler.running = False
            server_thread.join(timeout=1.0)
        server.close()


def test_replay_script_help_exposes_dataset_episode_and_timing_options():
    result = subprocess.run(
        [sys.executable, "scripts/replay_g1_zmq_policy.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--dataset-dir" in result.stdout
    assert "--episode-id" in result.stdout
    assert "--horizon" in result.stdout
    assert "--inference-delay-s" in result.stdout
