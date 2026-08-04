# G1 ZMQ Replay Policy Design

## Goal

Add a replay policy server that is a drop-in replacement for
`scripts/serve_g1_sonic_zmq_policy.py`. Instead of running OpenPI inference,
the server returns recorded G1 SONIC action trajectories from one episode of a
local LeRobot dataset.

The replay must reproduce the timing of the current end-to-end VLA path. The
new server adds 0.2 seconds of response delay, while the robot-side inference
worker retains its existing 0.2-second result hold. The expected total delay is
therefore about 0.4 seconds.

## Scope

- Add `scripts/replay_g1_zmq_policy.py` in the same directory as the existing
  ZMQ policy server.
- Default to `replay_data/walk_clean_desk2`, episode 0, while allowing both to
  be selected through CLI arguments.
- Read only these parquet columns:
  - `action.motion_token`, with 64 values per frame;
  - `teleop.left_hand_joints`, with 7 values per frame;
  - `teleop.right_hand_joints`, with 7 values per frame.
- Return an action horizon of 70 frames by default.
- Preserve the existing GR00T `PolicyClient` ZMQ and msgpack/NumPy protocol.
- Do not compare the recorded state with the live robot state or adjust the
  trajectory for pose differences.

## Command-Line Interface

The replay script exposes the following typed arguments:

- `dataset_dir`: dataset root, defaulting to
  `replay_data/walk_clean_desk2`;
- `episode_id`: episode parquet identifier, defaulting to 0;
- `host`: ZMQ bind host, defaulting to `0.0.0.0`;
- `port`: ZMQ REP port, defaulting to 29999;
- `horizon`: number of returned action frames, defaulting to 70;
- `inference_delay_s`: server-side artificial delay, defaulting to 0.2;
- `timeout_ms`: receive timeout, where 0 waits forever;
- `verbose_timing`: whether to log per-request replay timing.

For episode `N`, the parquet path is derived as
`data/chunk-{N // 1000:03d}/episode_{N:06d}.parquet`. The replay frequency is
read from `meta/info.json`; the current dataset declares 50 Hz.

## Architecture

### Dataset loading

At startup, a loader reads the dataset metadata and the three required parquet
columns. It converts each column into a contiguous `float32` NumPy array and
validates that:

- the metadata and parquet files exist;
- the metadata frequency is positive;
- the episode is non-empty;
- all columns have the same frame count;
- every motion token has width 64;
- every hand action has width 7;
- `horizon` is positive and `inference_delay_s` is non-negative.

Static data errors fail startup with a specific exception instead of appearing
after the robot begins replaying.

### Wall-clock replay timeline

The first successful `get_action` request establishes a monotonic-clock anchor
for dataset frame 0. For every request, the start frame is:

```text
round((request_time - anchor_time) * dataset_fps)
```

The request time is captured before the artificial delay. This makes the
returned chunk represent the trajectory from trigger time `t` through
`t + horizon / fps`, independent of the time at which the response reaches the
robot.

For example, at 50 Hz with horizon 70, requests received at elapsed times 0
and 1.0 seconds return frames `[0, 69]` and `[50, 119]`. The 20-frame overlap is
intentional and matches repeated VLA inference over overlapping future
windows. Indexing from the initial anchor, rather than incrementing a mutable
cursor, prevents request jitter from accumulating as replay drift.

If fewer than `horizon` recorded frames remain, the last recorded frame is
repeated until every returned array has the full horizon. Once the elapsed
time is beyond the episode, the complete chunk consists of the final frame.

The `reset` endpoint clears the anchor. The next successful `get_action`
request starts a new replay from frame 0.

### ZMQ server

The server uses the same REP loop, request envelope, msgpack/NumPy encoding,
and action validation as `serve_g1_sonic_zmq_policy.py`. It supports the same
core endpoints:

- `ping`: respond immediately with server status;
- `kill`: respond and stop the server loop;
- `reset`: clear the replay anchor and respond immediately;
- `get_metadata`: report dataset path, episode, frequency, episode length,
  horizon, and configured server delay;
- `get_action`: require `data["observation"]` for protocol compatibility, but
  deliberately ignore its contents.

For `get_action`, the server captures the request time, constructs the chunk,
waits `inference_delay_s`, validates the result, and returns `(action, info)`.
The action dictionary contains `motion_token`, `left_hand_joints`, and
`right_hand_joints`. The info dictionary reports the replay start frame,
replay time, server timing, and action shapes. Only successful action requests
incur the artificial delay; control endpoints remain responsive when the
server is not already processing a REP request.

Errors follow the existing server convention and are returned as
`{"error": "..."}` messages.

## Reuse and Isolation

The replay implementation reuses the existing server's `Gr00tMsgpack`, action
shape summary, and action validation helpers. Dataset loading and wall-clock
chunk selection remain independent of the socket loop so their behavior can be
tested without networking or real-time sleeps.

The clock and delay function are injectable in tests. Production defaults use
`time.monotonic` and `time.sleep`.

## Testing

Add `scripts/replay_g1_zmq_policy_test.py` with synthetic parquet data and
metadata under pytest's temporary directory. Tests cover:

1. loading the selected episode and converting the three actions to the
   expected `float32` shapes;
2. clear failures for a missing parquet, missing column, empty episode, or
   invalid action width;
3. the first request returning frames `[0, 69]`;
4. a request one second later returning `[50, 119]`, including the expected
   overlap with the first chunk;
5. final-frame padding near and beyond the end of an episode;
6. `reset` causing the next request to restart at frame 0;
7. capturing the replay index before applying the 0.2-second server delay;
8. protocol-compatible action keys, shapes, dtypes, and `(action, info)`
   response structure;
9. immediate behavior for `ping` and `reset` compared with delayed successful
   `get_action` handling; and
10. msgpack round trips for the action response.

Verification also includes the focused pytest module, Ruff checks for the new
script and test, Python compilation, and `git diff --check`.

## Success Criteria

- The script can select `walk_clean_desk2` or another compatible dataset and
  any available episode through CLI parameters.
- A request triggered at time `t` returns the 70 recorded frames beginning at
  the frame corresponding to `t`, then responds after about 0.2 seconds.
- Repeated requests use wall-clock positions and preserve overlapping frames.
- The existing robot-side 0.2-second hold remains unchanged, producing the
  requested approximate 0.4-second total delay.
- Every successful response remains shape-compatible at the end of an episode
  by repeating the last recorded frame.
- The server is usable by the existing GR00T `PolicyClient` without robot-side
  interface changes.
