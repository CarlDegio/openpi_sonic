import dataclasses

import einops
import numpy as np

from openpi import transforms


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
