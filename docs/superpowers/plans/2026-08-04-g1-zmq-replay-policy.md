# G1 ZMQ Replay Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a GR00T-compatible ZMQ server that replays one recorded G1 SONIC episode in overlapping, wall-clock-aligned 70-frame chunks with an additional 0.2-second server delay.

**Architecture:** A validated `ReplayEpisode` owns the three `float32` action arrays. A `ReplayTimeline` maps monotonic request times to padded chunks without an incrementing cursor, while a separately testable request handler applies endpoint semantics and delay; a thin REP loop provides the existing msgpack transport.

**Tech Stack:** Python 3.11, NumPy, Polars, pyzmq, msgpack, Tyro, pytest, Ruff.

## Global Constraints

- Create `scripts/replay_g1_zmq_policy.py`.
- Default to `replay_data/walk_clean_desk2`, episode 0, horizon 70, and exactly 0.2 seconds of server delay.
- Read fps from `meta/info.json`; current data is 50 Hz.
- Return `motion_token[T,64]`, `left_hand_joints[T,7]`, and `right_hand_joints[T,7]` as `float32`.
- Anchor frame 0 at the first successful `get_action`; later starts are `round((request_time - anchor_time) * fps)`.
- Preserve overlapping frames and repeat the last frame beyond the episode.
- `reset` clears the anchor; observation contents are ignored.
- Do not stage or modify the user's untracked `replay_data/`.

---

## File Structure

- `scripts/replay_g1_zmq_policy.py`: CLI, data loading, timeline, request handling, and ZMQ lifecycle.
- `scripts/replay_g1_zmq_policy_test.py`: synthetic-data unit and in-process transport tests.
- `docs/superpowers/plans/2026-08-04-g1-zmq-replay-policy.md`: execution checklist.

### Task 1: Episode Loading and Wall-Clock Chunking

**Files:**
- Create: `scripts/replay_g1_zmq_policy.py`
- Create: `scripts/replay_g1_zmq_policy_test.py`

**Interfaces:**
- Consumes: LeRobot `meta/info.json` and `data/chunk-{episode_id // 1000:03d}/episode_{episode_id:06d}.parquet`.
- Produces: `ReplayEpisode.load(dataset_dir: pathlib.Path, episode_id: int) -> ReplayEpisode` and `ReplayTimeline.get_action(request_time: float) -> tuple[dict[str, np.ndarray], int]`.

- [ ] **Step 1: Write failing loader tests**

Create a synthetic dataset fixture using this frame-identifiable data:

```python
ACTION_COLUMNS = {
    "motion_token": ("action.motion_token", 64),
    "left_hand_joints": ("teleop.left_hand_joints", 7),
    "right_hand_joints": ("teleop.right_hand_joints", 7),
}


def _write_episode(root: pathlib.Path, *, length: int = 130, fps: float = 50.0) -> pathlib.Path:
    (root / "meta").mkdir(parents=True)
    parquet_dir = root / "data" / "chunk-000"
    parquet_dir.mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"fps": fps}), encoding="utf-8")
    frames = np.arange(length, dtype=np.float32)
    pl.DataFrame(
        {
            column: [np.full(width, frame, dtype=np.float32).tolist() for frame in frames]
            for column, width in ACTION_COLUMNS.values()
        }
    ).write_parquet(parquet_dir / "episode_000000.parquet")
    return root
```

Assert exact `float32` shapes. Add individual tests for missing metadata, missing parquet, missing column, empty episode, wrong widths, non-positive fps, and negative episode id.

- [ ] **Step 2: Run loader tests and verify RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/openpi-sonic-uv-cache uv run pytest scripts/replay_g1_zmq_policy_test.py -k 'load or rejects' -v
```

Expected: collection fails because the replay module does not exist.

- [ ] **Step 3: Implement the validated loader**

Define:

```python
ACTION_COLUMNS = {
    "motion_token": ("action.motion_token", 64),
    "left_hand_joints": ("teleop.left_hand_joints", 7),
    "right_hand_joints": ("teleop.right_hand_joints", 7),
}


@dataclasses.dataclass(frozen=True)
class ReplayEpisode:
    dataset_dir: pathlib.Path
    episode_id: int
    fps: float
    actions: dict[str, np.ndarray]

    @property
    def length(self) -> int:
        return self.actions["motion_token"].shape[0]
```

`ReplayEpisode.load` must reject negative ids, read and validate positive `fps`, derive the chunk path from the id, inspect the parquet schema before selecting the three columns, stack each list column with `np.asarray(series.to_list(), dtype=np.float32)`, require exact `(length, width)` shapes, reject zero rows, and store `np.ascontiguousarray` results.

- [ ] **Step 4: Run loader tests and verify GREEN**

Repeat Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Write failing timeline tests**

Use the synthetic frame values to assert:

```python
timeline = ReplayTimeline(episode, horizon=70)
first, first_start = timeline.get_action(100.0)
second, second_start = timeline.get_action(101.0)

assert first_start == 0
assert second_start == 50
np.testing.assert_array_equal(first["motion_token"][50:], second["motion_token"][:20])
```

Add individual tests for final-row padding near the end, 70 final rows beyond the end, `reset()` restarting at frame 0, and a non-positive horizon error.

- [ ] **Step 6: Run timeline tests and verify RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/openpi-sonic-uv-cache uv run pytest scripts/replay_g1_zmq_policy_test.py -k 'timeline or overlap or padding or reset or horizon' -v
```

Expected: tests fail because `ReplayTimeline` is absent.

- [ ] **Step 7: Implement timeline selection**

Use this exact indexing rule:

```python
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
```

The slice helper clamps starts beyond the data to the final row, slices available frames, and appends `np.repeat(array[-1:], missing, axis=0)`.

- [ ] **Step 8: Run all Task 1 tests and commit**

Run the complete test module, then:

```bash
git add scripts/replay_g1_zmq_policy.py scripts/replay_g1_zmq_policy_test.py
git commit -m "feat: add wall-clock G1 action replay"
```

### Task 2: Endpoint Semantics, Delay, and Codec Compatibility

**Files:**
- Modify: `scripts/replay_g1_zmq_policy.py`
- Modify: `scripts/replay_g1_zmq_policy_test.py`

**Interfaces:**
- Consumes: `ReplayTimeline.get_action(request_time)` and existing `Gr00tMsgpack`, `_shape_summary`, and `validate_action`.
- Produces: `ReplayRequestHandler.handle(request: dict[str, Any]) -> Any` and `running: bool`.

- [ ] **Step 1: Write failing endpoint and timing tests**

Inject a fake clock whose sleeper records and advances time. Assert:

```python
action, info = handler.handle({"endpoint": "get_action", "data": {"observation": {}}})
assert fake_clock.sleeps == [0.2]
assert info["replay"]["start_frame"] == 0
assert info["server_timing"]["infer_ms"] == pytest.approx(200.0)
assert action["motion_token"].shape == (70, 64)
```

Advance the clock so the second request trigger is exactly one second after the first trigger and assert start frame 50, proving indexing happens before sleep. Add separate assertions that `ping`, `reset`, and `get_metadata` do not sleep; `reset` restarts replay; `kill` clears `running`; missing observation and unknown endpoints raise clear errors; and negative delay is rejected.

- [ ] **Step 2: Run handler tests and verify RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/openpi-sonic-uv-cache uv run pytest scripts/replay_g1_zmq_policy_test.py -k 'handler or endpoint or delay or metadata' -v
```

Expected: tests fail because `ReplayRequestHandler` is absent.

- [ ] **Step 3: Implement request handling**

Reuse the sibling server's validation helpers:

```python
from serve_g1_sonic_zmq_policy import _shape_summary
from serve_g1_sonic_zmq_policy import validate_action
```

For a successful action request, capture `request_time = clock()`, call the timeline, call `sleep(inference_delay_s)`, validate action width with a 1.25 warning threshold, and return:

```python
info = {
    "replay": {
        "episode_id": episode.episode_id,
        "start_frame": start_frame,
        "start_time_s": start_frame / episode.fps,
    },
    "server_timing": {"convert_ms": 0.0, "infer_ms": infer_ms},
    "action_shapes": _shape_summary(action),
}
```

Metadata contains dataset path, episode id, fps, length, horizon, and delay. Follow the existing responses for `ping`, `kill`, `reset`, `get_metadata`, and unknown endpoints. Require `data["observation"]` but do not inspect it.

- [ ] **Step 4: Run handler tests and verify GREEN**

Repeat Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Write a failing codec round-trip test**

Pack and unpack a handler response with `Gr00tMsgpack`; assert a two-item sequence and preserved action keys, dtypes, and shapes.

- [ ] **Step 6: Run codec test, integrate the codec, and verify GREEN**

Run:

```bash
env UV_CACHE_DIR=/tmp/openpi-sonic-uv-cache uv run pytest scripts/replay_g1_zmq_policy_test.py -k msgpack -v
```

The test must fail before codec integration. Then import
`Gr00tMsgpack` from `serve_g1_sonic_zmq_policy`, rerun the test to make it
pass, and run the full focused module.

- [ ] **Step 7: Commit Task 2**

```bash
git add scripts/replay_g1_zmq_policy.py scripts/replay_g1_zmq_policy_test.py
git commit -m "feat: emulate G1 replay inference timing"
```

### Task 3: ZMQ Lifecycle, CLI, and Final Verification

**Files:**
- Modify: `scripts/replay_g1_zmq_policy.py`
- Modify: `scripts/replay_g1_zmq_policy_test.py`

**Interfaces:**
- Consumes: Task 1–2 classes and `Gr00tMsgpack`.
- Produces: `G1ReplayZmqServer.serve_forever()`, `main(args: Args)`, and executable Tyro CLI.

- [ ] **Step 1: Write a failing in-process ZMQ test**

Start the server on loopback with an available port and zero delay. Through a real REQ socket, round-trip `ping`, `get_action`, `reset`, and `kill`. Assert action shapes `(70,64)`, `(70,7)`, `(70,7)`, and ensure the server thread exits after the kill response.

- [ ] **Step 2: Run ZMQ test and verify RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/openpi-sonic-uv-cache uv run pytest scripts/replay_g1_zmq_policy_test.py -k zmq -v
```

Expected: failure reports that `G1ReplayZmqServer` is absent.

- [ ] **Step 3: Implement the REP loop and CLI**

Define:

```python
@dataclasses.dataclass
class Args:
    dataset_dir: pathlib.Path = pathlib.Path("replay_data/walk_clean_desk2")
    episode_id: int = 0
    host: str = "0.0.0.0"
    port: int = 29999
    horizon: int = 70
    inference_delay_s: float = 0.2
    timeout_ms: int = 0
    verbose_timing: bool = False
```

Mirror the existing server's `LINGER=0`, optional `RCVTIMEO`, REP receive/unpack/handle/pack/send loop, and `{"error": str(exc)}` response convention. `close()` closes the socket and its owned context. `main` loads the episode, constructs the timeline and handler, logs selection metadata, serves until kill or Ctrl-C, and always closes.

- [ ] **Step 4: Run ZMQ test and verify GREEN**

Repeat Step 2. Expected: real socket round trips pass and the thread exits.

- [ ] **Step 5: Run focused and static verification**

Run:

```bash
env UV_CACHE_DIR=/tmp/openpi-sonic-uv-cache uv run pytest scripts/replay_g1_zmq_policy_test.py -v
env UV_CACHE_DIR=/tmp/openpi-sonic-uv-cache uv run ruff check scripts/replay_g1_zmq_policy.py scripts/replay_g1_zmq_policy_test.py
env UV_CACHE_DIR=/tmp/openpi-sonic-uv-cache uv run ruff format --check scripts/replay_g1_zmq_policy.py scripts/replay_g1_zmq_policy_test.py
env UV_CACHE_DIR=/tmp/openpi-sonic-uv-cache uv run python -m compileall -q scripts/replay_g1_zmq_policy.py scripts/replay_g1_zmq_policy_test.py
git diff --check -- scripts/replay_g1_zmq_policy.py scripts/replay_g1_zmq_policy_test.py
```

Expected: every command exits 0 without diagnostics.

- [ ] **Step 6: Smoke-test the real dataset**

Load `replay_data/walk_clean_desk2` episode 0 without binding a port and print fps, length, start frame, and shapes. Expected values include `50.0`, `2631`, `0`, `(70,64)`, `(70,7)`, and `(70,7)`.

- [ ] **Step 7: Review and commit only intended files**

Inspect `git diff --stat`, `git diff`, and `git status --short`. Confirm `replay_data/` remains untracked. Then:

```bash
git add docs/superpowers/plans/2026-08-04-g1-zmq-replay-policy.md scripts/replay_g1_zmq_policy.py scripts/replay_g1_zmq_policy_test.py
git commit -m "feat: add G1 ZMQ replay policy"
```
