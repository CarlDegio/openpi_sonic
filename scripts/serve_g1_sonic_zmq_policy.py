#!/usr/bin/env python3
"""Serve an OpenPI G1 SONIC policy over a GR00T-compatible ZMQ protocol.

This server is intended to be a drop-in replacement for the Isaac-GR00T
PolicyServer used by ``gear_sonic/scripts/run_vla_inference.py``.  The robot
machine can keep using GR00T's ``PolicyClient`` while the GPU machine runs an
OpenPI checkpoint.
"""

from __future__ import annotations

import dataclasses
import logging
import socket
import time
from typing import Any

import msgpack
import numpy as np
import tyro
import zmq


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    config: str
    """Training config name, e.g. pi05_g1_sonic_lora_movedoor."""

    dir: str
    """Checkpoint directory, e.g. checkpoints/pi05_g1_sonic_lora_movedoor/exp/29999."""


@dataclasses.dataclass
class Args:
    """Arguments for the G1 SONIC ZMQ policy server."""

    policy: Checkpoint
    """OpenPI checkpoint to serve."""

    host: str = "0.0.0.0"
    """Host/IP to bind."""

    port: int = 29999
    """ZMQ REP port to bind."""

    default_prompt: str | None = None
    """Fallback prompt used by OpenPI when the request does not include one."""

    timeout_ms: int = 0
    """Receive timeout in milliseconds. 0 means wait forever."""

    max_motion_token_abs: float = 1.25
    """Warn when predicted motion tokens exceed this absolute value."""

    verbose_timing: bool = False
    """Print per-request timing."""


class Gr00tMsgpack:
    """Minimal msgpack-numpy compatibility for GR00T PolicyClient messages."""

    @staticmethod
    def _get(obj: dict[Any, Any], key: str) -> Any:
        return obj.get(key, obj.get(key.encode("utf-8")))

    @staticmethod
    def encode(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind in ("O", "V"):
                raise ValueError(f"Unsupported ndarray dtype for msgpack: {obj.dtype}")
            return {
                b"nd": True,
                b"type": obj.dtype.str,
                b"kind": obj.dtype.kind,
                b"shape": obj.shape,
                b"data": obj.tobytes(),
            }
        if isinstance(obj, np.generic):
            return {
                b"nd": False,
                b"type": obj.dtype.str,
                b"data": obj.item(),
            }
        return obj

    @staticmethod
    def decode(obj: Any) -> Any:
        if not isinstance(obj, dict):
            return obj

        if b"nd" in obj or "nd" in obj:
            is_array = Gr00tMsgpack._get(obj, "nd")
            dtype = np.dtype(Gr00tMsgpack._get(obj, "type"))
            if is_array:
                return np.frombuffer(Gr00tMsgpack._get(obj, "data"), dtype=dtype).reshape(
                    Gr00tMsgpack._get(obj, "shape")
                )
            return np.asarray(Gr00tMsgpack._get(obj, "data"), dtype=dtype)[()]

        # Also accept OpenPI client's numpy-msgpack format. This is useful for
        # local tests and does not affect GR00T clients.
        if b"__ndarray__" in obj or "__ndarray__" in obj:
            return np.ndarray(
                buffer=Gr00tMsgpack._get(obj, "data"),
                dtype=np.dtype(Gr00tMsgpack._get(obj, "dtype")),
                shape=Gr00tMsgpack._get(obj, "shape"),
            )
        if b"__npgeneric__" in obj or "__npgeneric__" in obj:
            return np.dtype(Gr00tMsgpack._get(obj, "dtype")).type(Gr00tMsgpack._get(obj, "data"))

        return obj

    @classmethod
    def packb(cls, data: Any) -> bytes:
        return msgpack.packb(data, default=cls.encode, use_bin_type=True)

    @classmethod
    def unpackb(cls, data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=cls.decode, raw=False)


def _squeeze_bt(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim >= 2 and array.shape[0] == 1 and array.shape[1] == 1:
        return array[0, 0]
    if array.ndim >= 1 and array.shape[0] == 1:
        return array[0]
    raise ValueError(f"{name} expected leading batch/time dims [1, 1] or [1], got {array.shape}")


def _extract_prompt(observation: dict[str, Any]) -> str | None:
    language = observation.get("language", {})
    value = language.get("annotation.human.task_description")
    if value is None:
        value = observation.get("prompt")
    if value is None:
        return None

    while isinstance(value, (list, tuple)) and value:
        value = value[0]
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def _get_projected_gravity(observation: dict[str, Any]) -> np.ndarray:
    state = observation.get("state", {})
    if "projected_gravity" not in state:
        raise ValueError("Missing observation['state']['projected_gravity']")
    gravity = _squeeze_bt(state["projected_gravity"], name="state.projected_gravity")
    gravity = np.asarray(gravity, dtype=np.float32)
    if gravity.shape != (3,):
        raise ValueError(f"projected_gravity must have shape (3,), got {gravity.shape}")
    return gravity


def gr00t_observation_to_openpi(observation: dict[str, Any]) -> dict[str, Any]:
    """Convert GR00T-style observations into OpenPI G1 SONIC policy inputs."""
    video = observation.get("video")
    if not isinstance(video, dict):
        raise ValueError("Missing observation['video']")

    if "ego_view" not in video:
        raise ValueError("Missing observation['video']['ego_view']")
    images: dict[str, np.ndarray] = {
        "ego_view": np.asarray(_squeeze_bt(video["ego_view"], name="video.ego_view")),
    }

    if "left_wrist" in video:
        images["left_wrist"] = np.asarray(_squeeze_bt(video["left_wrist"], name="video.left_wrist"))
    if "wrist_view" in video:
        # OpenPI's G1SonicInputs performs the trained right-wrist 180 degree
        # rotation.  Do not rotate here.
        images["right_wrist"] = np.asarray(_squeeze_bt(video["wrist_view"], name="video.wrist_view"))
    elif "right_wrist" in video:
        images["right_wrist"] = np.asarray(_squeeze_bt(video["right_wrist"], name="video.right_wrist"))

    if "q" not in observation:
        raise ValueError("Missing observation['q']")
    state = np.asarray(_squeeze_bt(observation["q"], name="q"), dtype=np.float32)
    if state.shape != (43,):
        raise ValueError(f"G1 SONIC observation state must have shape (43,), got {state.shape}")

    openpi_obs: dict[str, Any] = {
        "images": images,
        "observation": {
            "state": state,
            "projected_gravity": _get_projected_gravity(observation),
        },
    }

    prompt = _extract_prompt(observation)
    if prompt is not None:
        openpi_obs["prompt"] = prompt

    return openpi_obs


def create_policy(args: Args) -> Any:
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    return _policy_config.create_trained_policy(
        _config.get_config(args.policy.config),
        args.policy.dir,
        default_prompt=args.default_prompt,
    )


def _shape_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: list(np.asarray(value).shape)
        for key, value in result.items()
        if key in {"motion_token", "left_hand_joints", "right_hand_joints", "actions"}
    }


def validate_action(result: dict[str, Any], *, max_motion_token_abs: float) -> None:
    required = ("motion_token", "left_hand_joints", "right_hand_joints")
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(f"Policy result missing keys {missing}; available keys: {list(result.keys())}")

    motion_token = np.asarray(result["motion_token"], dtype=np.float32)
    left_hand = np.asarray(result["left_hand_joints"], dtype=np.float32)
    right_hand = np.asarray(result["right_hand_joints"], dtype=np.float32)

    if motion_token.ndim != 2 or motion_token.shape[-1] != 64:
        raise ValueError(f"motion_token must have shape [T, 64], got {motion_token.shape}")
    if left_hand.shape != (motion_token.shape[0], 7):
        raise ValueError(f"left_hand_joints must have shape [{motion_token.shape[0]}, 7], got {left_hand.shape}")
    if right_hand.shape != (motion_token.shape[0], 7):
        raise ValueError(f"right_hand_joints must have shape [{motion_token.shape[0]}, 7], got {right_hand.shape}")

    max_abs = float(np.abs(motion_token).max()) if motion_token.size else 0.0
    if max_abs > max_motion_token_abs:
        logging.warning(
            "motion_token max abs %.4f exceeds configured warning threshold %.4f",
            max_abs,
            max_motion_token_abs,
        )


class OpenPIG1SonicZmqServer:
    def __init__(self, args: Args, policy: Any) -> None:
        self._args = args
        self._policy = policy
        self._running = True
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        if args.timeout_ms > 0:
            self._socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)

        endpoint = f"tcp://{args.host}:{args.port}"
        self._socket.bind(endpoint)
        self._endpoint = endpoint

    def close(self) -> None:
        self._socket.close()
        self._context.term()

    def _handle_request(self, request: dict[str, Any]) -> Any:
        endpoint = request.get("endpoint", "get_action")

        if endpoint == "ping":
            return {"status": "ok", "message": "OpenPI G1 SONIC ZMQ server is running"}
        if endpoint == "kill":
            self._running = False
            return {"status": "ok", "message": "Server shutting down"}
        if endpoint == "reset":
            return {"status": "ok"}
        if endpoint == "get_metadata":
            return self._policy.metadata
        if endpoint != "get_action":
            raise ValueError(f"Unknown endpoint: {endpoint}")

        data = request.get("data", {})
        if not isinstance(data, dict) or "observation" not in data:
            raise ValueError("get_action request must include data['observation']")

        convert_start = time.monotonic()
        openpi_obs = gr00t_observation_to_openpi(data["observation"])
        convert_ms = (time.monotonic() - convert_start) * 1000

        infer_start = time.monotonic()
        action = self._policy.infer(openpi_obs)
        infer_ms = (time.monotonic() - infer_start) * 1000

        validate_action(action, max_motion_token_abs=self._args.max_motion_token_abs)

        info = {
            "server_timing": {
                "convert_ms": convert_ms,
                "infer_ms": infer_ms,
            },
            "action_shapes": _shape_summary(action),
        }
        if self._args.verbose_timing:
            logging.info(
                "request done convert=%.2fms infer=%.2fms shapes=%s",
                convert_ms,
                infer_ms,
                info["action_shapes"],
            )
        return action, info

    def serve_forever(self) -> None:
        logging.info("Server ready on %s", self._endpoint)
        while self._running:
            try:
                raw = self._socket.recv()
                request = Gr00tMsgpack.unpackb(raw)
                result = self._handle_request(request)
                self._socket.send(Gr00tMsgpack.packb(result))
            except zmq.Again:
                continue
            except Exception as exc:
                logging.exception("Error while handling request")
                self._socket.send(Gr00tMsgpack.packb({"error": str(exc)}))


def main(args: Args) -> None:
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Loading OpenPI policy config=%s dir=%s", args.policy.config, args.policy.dir)
    policy = create_policy(args)
    logging.info("Policy loaded on host=%s ip=%s", hostname, local_ip)

    server = OpenPIG1SonicZmqServer(args, policy)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Interrupted, shutting down")
    finally:
        server.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
