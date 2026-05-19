"""Evaluate a G1 SONIC pi0.5 policy server against a local LeRobot episode."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

import numpy as np
from openpi_client import msgpack_numpy
import pandas as pd
from tqdm.auto import tqdm
import tyro
import websockets.sync.client

from openpi.training import config as _config

logger = logging.getLogger(__name__)


ACTION_KEYS = ("action.motion_token", "teleop.left_hand_joints", "teleop.right_hand_joints")
PLOT_GROUPS = (
    *[(f"action_dims_{start + 1:03d}_{start + 8:03d}", start, start + 8) for start in range(0, 64, 8)],
    ("left_hand_065_071", 64, 71),
    ("right_hand_072_078", 71, 78),
)


@dataclasses.dataclass
class Args:
    """Arguments for G1 SONIC offline policy evaluation."""

    # Local LeRobot dataset path, e.g. /mnt/g1_training_dataset/MoveDoorMerge.
    dataset_path: pathlib.Path = pathlib.Path("/mnt/g1_training_dataset/MoveDoorMerge")
    # Episode index inside the LeRobot dataset.
    episode_index: int = 0
    # Directory for plots, metrics, and optional arrays.
    output_dir: pathlib.Path = pathlib.Path("outputs/movedoor_eval")

    # Policy server host.
    host: str = "localhost"
    # Policy server port.
    port: int = 8000
    # Seconds to wait for the policy server connection.
    connect_timeout_s: float = 300.0

    # If true, this script starts scripts/serve_policy.py before evaluating.
    start_server: bool = False
    # Config passed to scripts/serve_policy.py when --start-server is enabled.
    checkpoint_config: str = "pi05_g1_sonic_lora_movedoor"
    # Checkpoint dir passed to scripts/serve_policy.py when --start-server is enabled.
    checkpoint_dir: pathlib.Path = pathlib.Path("checkpoints/pi05_g1_sonic_lora_movedoor/movedoor_lora/29999")

    # Number of predicted actions per policy call. Must match the trained model horizon.
    chunk_size: int = 40
    # Target-frame offset relative to each observation frame. Use 0 to match training chunks [t, t+39].
    target_offset: int = 0
    # Optional cap on the number of target frames to evaluate.
    max_frames: int | None = None
    # Optional prompt override. If set, task_index is omitted so this prompt is used by the policy transform.
    prompt: str | None = None
    # Save pred_actions.npy, gt_actions.npy, and frame_indices.npy.
    save_npy: bool = True

    # Cache directory used by HuggingFace datasets if HF_DATASETS_CACHE is not already set.
    hf_datasets_cache: pathlib.Path = pathlib.Path("/tmp/hf_datasets")
    # HF_HOME used if HF_HOME is not already set.
    hf_home: pathlib.Path = pathlib.Path("/tmp/hf_home")


class PolicyClient:
    """Small websocket client with a bounded connection wait."""

    def __init__(self, host: str, port: int, *, timeout_s: float) -> None:
        uri = host if host.startswith("ws") else f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self.metadata = self._connect(uri, timeout_s)

    def _connect(self, uri: str, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                conn = websockets.sync.client.connect(uri, compression=None, max_size=None, open_timeout=5)
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except OSError as exc:
                last_error = exc
                time.sleep(2)
        raise TimeoutError(f"Timed out waiting for policy server at {uri}") from last_error

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Policy server returned an error:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        self._ws.close()


def _tail_text(path: pathlib.Path, *, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _configure_dataset_env(dataset_path: pathlib.Path, args: Args) -> tuple[str, pathlib.Path]:
    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    os.environ["HF_LEROBOT_HOME"] = str(dataset_path.parent)
    os.environ.setdefault("HF_HOME", str(args.hf_home.expanduser().resolve()))
    os.environ.setdefault("HF_DATASETS_CACHE", str(args.hf_datasets_cache.expanduser().resolve()))
    pathlib.Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
    return dataset_path.name, dataset_path


def _episode_bounds(dataset_path: pathlib.Path, episode_index: int) -> tuple[int, int]:
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing episode metadata: {episodes_path}")

    start = 0
    with episodes_path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            length = int(item["length"])
            if int(item["episode_index"]) == episode_index:
                return start, length
            start += length
    raise ValueError(f"Episode {episode_index} was not found in {episodes_path}")


def _read_episode_frame(dataset_path: pathlib.Path, episode_index: int) -> pd.DataFrame:
    parquet_path = dataset_path / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing episode parquet: {parquet_path}")
    return pd.read_parquet(parquet_path)


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _to_numpy(item) for key, item in value.items()}
    return value


def _make_observation(sample: dict[str, Any], prompt: str | None) -> dict[str, Any]:
    sample = _to_numpy(sample)
    obs: dict[str, Any] = {
        "images": {
            "ego_view": sample["observation.images.ego_view"],
            "left_wrist": sample["observation.images.left_wrist"],
            "right_wrist": sample["observation.images.right_wrist"],
        },
        "observation": {
            "state": sample["observation.state"],
            "projected_gravity": sample["observation.projected_gravity"],
        },
    }
    if prompt is None:
        obs["task_index"] = sample["task_index"]
        if "task" in sample:
            obs["prompt"] = sample["task"]
    else:
        obs["prompt"] = prompt
    return obs


def _concat_policy_actions(result: dict[str, Any]) -> np.ndarray:
    if "actions" in result:
        actions = np.asarray(result["actions"], dtype=np.float32)
    else:
        actions = np.concatenate(
            [
                np.asarray(result["motion_token"], dtype=np.float32),
                np.asarray(result["left_hand_joints"], dtype=np.float32),
                np.asarray(result["right_hand_joints"], dtype=np.float32),
            ],
            axis=-1,
        )
    if actions.ndim != 2:
        raise ValueError(f"Expected policy actions to be rank 2, got shape {actions.shape}")
    return actions


def _concat_ground_truth(frame: pd.DataFrame, start: int, end: int) -> np.ndarray:
    columns = [np.stack(frame[key].iloc[start:end].to_numpy()).astype(np.float32) for key in ACTION_KEYS]
    return np.concatenate(columns, axis=-1)


def _validate_config(args: Args) -> None:
    train_config = _config.get_config(args.checkpoint_config)
    if train_config.model.action_horizon != args.chunk_size:
        raise ValueError(
            f"--chunk-size={args.chunk_size} does not match config action_horizon="
            f"{train_config.model.action_horizon}"
        )
    if train_config.model.action_dim != 78:
        raise ValueError(f"Expected G1 SONIC action_dim=78, got {train_config.model.action_dim}")


def _start_policy_server(args: Args) -> tuple[subprocess.Popen, Any]:
    log_path = args.output_dir / "policy_server.log"
    log_file = log_path.open("w", encoding="utf-8")
    command = [
        sys.executable,
        "scripts/serve_policy.py",
        f"--port={args.port}",
        "policy:checkpoint",
        f"--policy.config={args.checkpoint_config}",
        f"--policy.dir={args.checkpoint_dir}",
    ]
    logger.info("Starting policy server; logs will be written to %s", log_path)
    process = subprocess.Popen(command, cwd=pathlib.Path(__file__).resolve().parents[1], stdout=log_file, stderr=log_file)
    return process, log_file


def _evaluate(args: Args, policy: PolicyClient, repo_id: str, dataset_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    global_start, episode_len = _episode_bounds(dataset_path, args.episode_index)
    frame = _read_episode_frame(dataset_path, args.episode_index)
    if len(frame) != episode_len:
        raise ValueError(f"Episode metadata length {episode_len} does not match parquet length {len(frame)}")

    dataset = LeRobotDataset(repo_id)
    max_target_len = episode_len - args.target_offset
    if args.max_frames is not None:
        max_target_len = min(max_target_len, args.max_frames)
    if max_target_len <= 0:
        raise ValueError("No frames to evaluate; check --target-offset and --max-frames")

    pred_chunks: list[np.ndarray] = []
    gt_chunks: list[np.ndarray] = []
    frame_index_chunks: list[np.ndarray] = []

    chunk_starts = range(0, max_target_len, args.chunk_size)
    for chunk_start in tqdm(chunk_starts, desc=f"Episode {args.episode_index} chunks"):
        obs_frame = chunk_start
        target_start = chunk_start + args.target_offset
        target_end = min(target_start + args.chunk_size, args.target_offset + max_target_len, episode_len)
        sample = dataset[global_start + obs_frame]
        result = policy.infer(_make_observation(sample, args.prompt))
        pred = _concat_policy_actions(result)[: target_end - target_start]
        gt = _concat_ground_truth(frame, target_start, target_end)

        if pred.shape != gt.shape:
            raise ValueError(f"Prediction shape {pred.shape} does not match ground truth shape {gt.shape}")
        if pred.shape[-1] != 78:
            raise ValueError(f"Expected 78 action dimensions, got {pred.shape[-1]}")

        pred_chunks.append(pred)
        gt_chunks.append(gt)
        frame_index_chunks.append(frame["frame_index"].iloc[target_start:target_end].to_numpy(dtype=np.int64))

    return np.concatenate(pred_chunks), np.concatenate(gt_chunks), np.concatenate(frame_index_chunks)


def _compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "all_dims": {
            "mae": float(np.mean(np.abs(pred - gt))),
            "rmse": float(np.sqrt(np.mean(np.square(pred - gt)))),
        },
        "groups": {},
    }
    for name, start, end in PLOT_GROUPS:
        group_pred = pred[:, start:end]
        group_gt = gt[:, start:end]
        metrics["groups"][name] = {
            "dims_1_based": [start + 1, end],
            "mae": float(np.mean(np.abs(group_pred - group_gt))),
            "rmse": float(np.sqrt(np.mean(np.square(group_pred - group_gt)))),
        }
    return metrics


def _write_plots(
    pred: np.ndarray,
    gt: np.ndarray,
    frame_indices: np.ndarray,
    output_dir: pathlib.Path,
    *,
    chunk_size: int,
) -> np.ndarray:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    pathlib.Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    x_values = frame_indices + 1
    input_marker_indices = np.arange(0, len(x_values), chunk_size)
    input_frame_numbers = x_values[input_marker_indices]

    def draw_group(ax, name: str, start: int, end: int) -> None:
        pred_mean = pred[:, start:end].mean(axis=-1)
        gt_mean = gt[:, start:end].mean(axis=-1)
        ax.plot(x_values, gt_mean, label="actual mean", linewidth=1.6)
        ax.plot(x_values, pred_mean, label="predicted mean", linewidth=1.4, alpha=0.9)
        ax.scatter(
            input_frame_numbers,
            gt_mean[input_marker_indices],
            label="input frames",
            color="black",
            s=16,
            zorder=4,
        )
        ax.scatter(
            input_frame_numbers,
            pred_mean[input_marker_indices],
            facecolors="white",
            edgecolors="black",
            s=22,
            linewidths=0.9,
            zorder=4,
        )
        ax.set_title(f"{name} (dims {start + 1}-{end})")
        ax.set_xlabel("episode frame (1-based)")
        ax.set_ylabel("mean action value")
        ax.grid(visible=True, linewidth=0.4, alpha=0.35)

    combined_fig, axes = plt.subplots(5, 2, figsize=(18, 22), dpi=150, sharex=True)
    for ax, (name, start, end) in zip(axes.ravel(), PLOT_GROUPS, strict=True):
        draw_group(ax, name, start, end)
        ax.legend(loc="best", fontsize=8)
    combined_fig.tight_layout()
    combined_fig.savefig(output_dir / "all_action_groups_5x2.png")
    plt.close(combined_fig)
    return input_frame_numbers


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo_id, dataset_path = _configure_dataset_env(args.dataset_path, args)
    _validate_config(args)

    server_process: subprocess.Popen | None = None
    server_log_file = None
    policy: PolicyClient | None = None
    try:
        if args.start_server:
            server_process, server_log_file = _start_policy_server(args)

        if server_process is not None and server_process.poll() is not None:
            log_tail = _tail_text(args.output_dir / "policy_server.log")
            raise RuntimeError(
                f"Policy server exited early with code {server_process.returncode}.\n"
                f"Log tail:\n{log_tail}"
            )

        try:
            policy = PolicyClient(args.host, args.port, timeout_s=args.connect_timeout_s)
        except TimeoutError:
            if server_process is not None and server_process.poll() is not None:
                log_tail = _tail_text(args.output_dir / "policy_server.log")
                raise RuntimeError(
                    f"Policy server exited with code {server_process.returncode} before accepting connections.\n"
                    f"Log tail:\n{log_tail}"
                ) from None
            raise
        logger.info("Connected to policy server with metadata: %s", policy.metadata)
        pred, gt, frame_indices = _evaluate(args, policy, repo_id, dataset_path)

        input_frame_numbers = _write_plots(pred, gt, frame_indices, args.output_dir, chunk_size=args.chunk_size)
        metrics = _compute_metrics(pred, gt)
        metrics.update(
            {
                "dataset_path": str(dataset_path),
                "episode_index": args.episode_index,
                "num_frames": int(pred.shape[0]),
                "action_dim": int(pred.shape[1]),
                "chunk_size": args.chunk_size,
                "target_offset": args.target_offset,
                "input_frame_numbers_1_based": input_frame_numbers.tolist(),
            }
        )
        (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        if args.save_npy:
            np.save(args.output_dir / "pred_actions.npy", pred)
            np.save(args.output_dir / "gt_actions.npy", gt)
            np.save(args.output_dir / "frame_indices.npy", frame_indices)

        logger.info("Wrote one 5x2 combined plot and metrics to %s", args.output_dir)
    finally:
        if policy is not None:
            policy.close()
        if server_process is not None:
            server_process.terminate()
            try:
                server_process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait()
        if server_log_file is not None:
            server_log_file.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
