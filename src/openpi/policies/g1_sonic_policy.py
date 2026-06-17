import dataclasses

import einops
import numpy as np

from openpi import transforms

SONIC_REAL_ACTION_DIM = 78
SONIC_MOTION_TOKEN_DIM = 64
SONIC_HAND_DIM = 7
PI05_ACTION_DIM = 32
SONIC_PACK_FACTOR = 3


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class G1SonicInputs(transforms.DataTransformFn):
    """Inputs for Unitree G1 SONIC VLA datasets.

    Expected training data after repacking:
    - images/ego_view, images/chest_view: RGB body-mounted images.
    - images/left_wrist, images/right_wrist: RGB wrist images.
    - observation/state: 43-D G1 joint configuration.
    - observation/projected_gravity: 3-D projected gravity vector.
    - action/motion_token: [T, 64] SONIC latent action chunk.
    - teleop/left_hand_joints, teleop/right_hand_joints: [T, 7] hand action chunks.
    """

    task_instruction_variants: dict[int, tuple[str, ...]] = dataclasses.field(default_factory=dict)

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        image_sources = {
            "base_0_rgb": "ego_view",
            "chest_0_rgb": "chest_view",
            "left_wrist_0_rgb": "left_wrist",
            "right_wrist_0_rgb": "right_wrist",
        }
        missing_images = [source for source in image_sources.values() if source not in in_images]
        if missing_images:
            raise ValueError(f"Missing required G1 SONIC camera views: {missing_images}")

        images = {}
        image_masks = {}
        for dest, source in image_sources.items():
            image = _parse_image(in_images[source])
            if source == "right_wrist":
                image = np.rot90(image, 2)
            images[dest] = image
            image_masks[dest] = np.True_

        observation = data["observation"]
        state = np.concatenate(
            [
                np.asarray(observation["state"]),
                np.asarray(observation["projected_gravity"]),
            ],
            axis=-1,
        )

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "action" in data:
            inputs["actions"] = np.concatenate(
                [
                    np.asarray(data["action"]["motion_token"]),
                    np.asarray(data["teleop"]["left_hand_joints"]),
                    np.asarray(data["teleop"]["right_hand_joints"]),
                ],
                axis=-1,
            )

        if "task_index" in data:
            variants = self.task_instruction_variants.get(int(data["task_index"]))
            if variants:
                inputs["prompt"] = np.random.choice(variants)

        if "prompt" in data and "prompt" not in inputs:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class G1SonicOutputs(transforms.DataTransformFn):
    """Split OpenPI actions back into SONIC latent protocol fields."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {
            "motion_token": actions[..., :64],
            "left_hand_joints": actions[..., 64:71],
            "right_hand_joints": actions[..., 71:78],
        }


@dataclasses.dataclass(frozen=True)
class PackG1SonicPi05Actions(transforms.DataTransformFn):
    """Pack normalized SONIC actions into pi0.5's native 32-D action tokens."""

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data

        actions = np.asarray(data["actions"])
        if actions.shape[-1] != SONIC_REAL_ACTION_DIM:
            raise ValueError(f"G1 SONIC actions must have last dim {SONIC_REAL_ACTION_DIM}, got {actions.shape}")

        motion_token = actions[..., :SONIC_MOTION_TOKEN_DIM]
        left_hand = actions[..., SONIC_MOTION_TOKEN_DIM : SONIC_MOTION_TOKEN_DIM + SONIC_HAND_DIM]
        right_hand = actions[..., SONIC_MOTION_TOKEN_DIM + SONIC_HAND_DIM : SONIC_REAL_ACTION_DIM]

        packed_shape = (*actions.shape[:-2], actions.shape[-2] * SONIC_PACK_FACTOR, PI05_ACTION_DIM)
        packed = np.zeros(packed_shape, dtype=actions.dtype)
        packed[..., 0::SONIC_PACK_FACTOR, :] = motion_token[..., :PI05_ACTION_DIM]
        packed[..., 1::SONIC_PACK_FACTOR, :] = motion_token[..., PI05_ACTION_DIM:SONIC_MOTION_TOKEN_DIM]
        packed[..., 2::SONIC_PACK_FACTOR, :SONIC_HAND_DIM] = left_hand
        packed[..., 2::SONIC_PACK_FACTOR, SONIC_HAND_DIM : 2 * SONIC_HAND_DIM] = right_hand

        action_loss_mask = np.ones(packed_shape, dtype=bool)
        action_loss_mask[..., 2::SONIC_PACK_FACTOR, 2 * SONIC_HAND_DIM :] = False

        return {
            **data,
            "actions": packed,
            "action_loss_mask": action_loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class UnpackG1SonicPi05Actions(transforms.DataTransformFn):
    """Unpack pi0.5 32-D action tokens back into normalized SONIC actions."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] != PI05_ACTION_DIM:
            raise ValueError(f"Packed G1 SONIC actions must have last dim {PI05_ACTION_DIM}, got {actions.shape}")
        if actions.shape[-2] % SONIC_PACK_FACTOR != 0:
            raise ValueError(
                f"Packed G1 SONIC action horizon must be divisible by {SONIC_PACK_FACTOR}, got {actions.shape}"
            )

        horizon = actions.shape[-2] // SONIC_PACK_FACTOR
        unpacked_shape = (*actions.shape[:-2], horizon, SONIC_REAL_ACTION_DIM)
        unpacked = np.zeros(unpacked_shape, dtype=actions.dtype)
        unpacked[..., :PI05_ACTION_DIM] = actions[..., 0::SONIC_PACK_FACTOR, :]
        unpacked[..., PI05_ACTION_DIM:SONIC_MOTION_TOKEN_DIM] = actions[..., 1::SONIC_PACK_FACTOR, :]
        unpacked[..., SONIC_MOTION_TOKEN_DIM : SONIC_MOTION_TOKEN_DIM + SONIC_HAND_DIM] = actions[
            ..., 2::SONIC_PACK_FACTOR, :SONIC_HAND_DIM
        ]
        unpacked[..., SONIC_MOTION_TOKEN_DIM + SONIC_HAND_DIM : SONIC_REAL_ACTION_DIM] = actions[
            ..., 2::SONIC_PACK_FACTOR, SONIC_HAND_DIM : 2 * SONIC_HAND_DIM
        ]

        return {
            **data,
            "actions": unpacked,
        }
