# G1 Replay Pause and Resume Design

## Goal

Synchronize recorded replay with the robot-side POSE and pause controls:

- no inference or observation transmission before POSE mode is ready and the
  policy loop is resumed;
- pressing `I` starts a new replay episode at frame 0 while remaining paused;
- pressing `P` pauses at the last action frame actually published to the
  robot;
- pressing `P` again resumes from that frozen frame, excluding paused wall
  time from replay progression.

The change spans the OpenPI replay server and
`GR00T-WholeBodyControl/gear_sonic/scripts/run_vla_inference.py`.
`launch_inference.py` remains unchanged because its keyboard process already
forwards raw `i` and `p` messages correctly.

## Existing Behavior and Root Cause

`launch_inference.py` publishes keyboard strings on port 5580. The actual
state transitions are handled by `run_vla_inference.py`.

The inference loop currently consumes and schedules inference before checking
`cpp_mode` or `pause_loop`. Consequently, its initial `pause_loop=True`
does not prevent sensor collection or `get_action` requests. The replay server
anchors frame 0 on the first such request, often before the operator presses
`I` or `P`.

The current key handlers only modify local state:

- `I` resets the local ZMQ frame counter, cached chunk, and chunk index;
- `P` toggles `pause_loop`;
- neither key calls `PolicyClient.reset()`.

The full observation is currently captured and transmitted during these early
requests. The replay handler only checks that `data["observation"]` exists;
it deliberately ignores image and state values because actions come from the
recorded episode.

## Protocol Choice

Use the existing GR00T reset endpoint instead of adding custom pause and resume
endpoints.

`PolicyClient.reset(options={"step_index": N})` is already part of the GR00T
interface. NVIDIA's `ReplayPolicy` implements this option by assigning
`current_step=N`; `Gr00tPolicy` safely ignores the options. The existing
OpenPI model server accepts reset and returns success without policy state.

The OpenPI replay server will implement the same `step_index` option. This
keeps one control path compatible with:

- the OpenPI replay server;
- NVIDIA's GR00T ReplayPolicy;
- the current OpenPI model server;
- GR00T model policies.

## Operator State Machine

### Startup

The inference process starts with `pause_loop=True` and C++ mode `OFF`.
Inference is inactive. It does not read cameras or robot state and does not
send `get_action`.

### K: start control

`K` starts the C++ loop in PLANNER mode. Inference remains inactive because
the mode is not POSE.

### I: initialize POSE and a new replay

`I` performs the existing initial-pose ramp, switches the C++ loop to POSE,
and forces `pause_loop=True`. It then:

1. clears cached action state and queued inference/results;
2. resets the tracked replay frame to 0;
3. calls `PolicyClient.reset(options={"step_index": 0})`.

A reset failure leaves the loop paused. A successful reset leaves it ready but
does not trigger inference. The first post-`I` `P` press is the start
trigger.

### P: resume

When POSE is active and the loop is paused, `P`:

1. keeps inference disabled while state is being changed;
2. discards cached chunks, queued requests, and stale completed results;
3. calls `PolicyClient.reset(options={"step_index": N})`, where `N` is the
   last replay frame actually published, or 0 if no replay frame has been
   published;
4. changes to the resumed state only after reset succeeds.

The following loop iteration captures a fresh observation and sends the first
new `get_action`. The returned chunk starts at frame `N`; repeating the
held frame at the boundary is intentional.

### P: pause

When POSE is active and the loop is running, `P` changes to paused
immediately from the publishing loop's perspective. It:

1. stops scheduling new inference;
2. stops publishing actions;
3. drains queued requests and discards results that finish after the pause;
4. retains the last replay frame actually published as `N`.

No policy request is required at pause time. Resetting to `N` on the later
resume corrects for any in-flight server request and removes all paused wall
time from the OpenPI replay timeline.

Pressing `P` outside POSE produces a warning and does not change replay
state.

### O or C++ OFF

PLANNER and OFF modes are inactive inference states. No observation is
captured and no `get_action` is scheduled. Returning through `I` creates a
new replay from frame 0.

## Tracking the Executed Replay Frame

The OpenPI replay response already reports
`info["replay"]["start_frame"]`. NVIDIA's ReplayPolicy reports
`info["current_step"]`. Robot-side inference processing will preserve policy
info instead of discarding it.

When a new chunk becomes active, the client records its replay start frame.
Whenever it publishes action index `current_idx`, it updates:

```text
last_replay_frame = chunk_start_frame + current_idx
```

If a policy response has neither replay field, replay-frame tracking remains
unavailable. Reset calls still work for model policies because their reset
implementations ignore `step_index`.

## OpenPI Replay Reset Semantics

`ReplayTimeline` gains a frame offset. Reset accepts a non-negative
`step_index`, clears the monotonic anchor, and assigns the offset. The next
successful action request anchors wall time at that offset:

```text
start_frame = reset_step_index + round((request_time - anchor_time) * fps)
```

Reset without options remains equivalent to `step_index=0`. Negative or
non-integer step indices are rejected. Values beyond the episode retain the
existing final-frame padding behavior.

The replay reset endpoint reads the standard GR00T request shape:

```python
{"endpoint": "reset", "data": {"options": {"step_index": N}}}
```

## Concurrency and Stale Results

The GR00T PolicyClient uses one ZMQ REQ socket, which must not be used
concurrently. A shared lock serializes `get_action` and `reset` calls.

A key transition may wait for an in-flight server response before issuing
reset, but it stops action publication immediately. The explicit
`step_index=N` then restores the server to the robot's last executed replay
frame, so server processing time cannot advance the resumed position.

The worker's existing artificial 0.2-second result hold is retained. Results
released during an inactive state are drained and never become the resumed
cached chunk.

## Observation Behavior

`prepare_observation_from_sensors` remains unchanged. It continues to build
and send the real four-camera, joint-state, projected-gravity, language, and
timestamp observation whenever active inference is requested.

The scheduling gate is evaluated before queueing worker work:

```text
active = cpp_loop_running and cpp_mode == "POSE" and not pause_loop
```

When inactive, the worker receives no new inference item, so it does not read
sensors or transmit an observation. Replay continues to ignore observation
values after validating the request envelope.

## Error Handling

- If reset at `I` or Resume fails, remain paused, clear stale state, and print
  the server error.
- Reject malformed replay reset data with a descriptive response.
- Reject negative or non-integral `step_index`.
- Keep reset idempotent.
- Preserve current request error encoding and socket recovery behavior.
- Preserve unrelated working-tree changes in both repositories. In particular,
  do not modify the existing dirty `launch_inference.py`,
  `reasan_planner.py`, LaViRA tests, or untracked planning document.

## Testing

### OpenPI repository

Extend `scripts/replay_g1_zmq_policy_test.py` to verify:

1. reset without options reanchors at frame 0;
2. reset with `step_index=N` makes the next chunk start at `N`;
3. later requests advance by wall-clock time relative to `N`;
4. reset during an established timeline removes elapsed paused time;
5. malformed, negative, and non-integral step indices fail clearly;
6. reset request encoding matches GR00T PolicyClient's `data.options` shape;
7. final-frame padding still works after an offset reset.

### GR00T-WholeBodyControl repository

Extend the focused inference tests to verify:

1. inactive startup, PLANNER, OFF, and pause states cannot schedule inference;
2. active POSE state can schedule inference;
3. policy info preserves both OpenPI `replay.start_frame` and NVIDIA
   `current_step`;
4. the last published replay frame is derived from chunk start plus action
   index;
5. stale inference and result queues are drained on key transitions;
6. reset is serialized with inference calls;
7. reset failure leaves inference paused;
8. the existing 0.2-second worker hold and latency compensation still pass.

Run repository-focused tests, Ruff/format/compile checks, and an in-process
client/server test covering `reset(step_index) → get_action → pause gap →
reset(step_index) → get_action`.

## Success Criteria

- Starting `launch_inference.py` does not produce policy action requests
  before the operator completes `I` and resumes with `P`.
- `I` resets replay to frame 0 and leaves inference paused.
- `P` pause stops observation transmission and action publication.
- `P` resume returns to the last replay frame actually published, regardless
  of pause duration or an in-flight inference.
- Normal active inference still sends the complete live observation.
- Existing OpenPI and GR00T model servers remain usable through the standard
  reset interface.

