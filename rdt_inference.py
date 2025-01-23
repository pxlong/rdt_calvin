"""Script to run a trained policy from RDT."""

"""Launch Isaac Sim Simulator first."""

import torch
import numpy as np
import yaml
from PIL import Image as PImage
from collections import deque

import os
import sys

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将上两层目录添加到 sys.path
sys.path.append(current_dir)
print(f"current_dir: {current_dir}")

from calvin_agent.models.calvin_base_model import CalvinBaseModel

from rdt_model import create_model


class RDTCalvinEvaluation(CalvinBaseModel):
    def __init__(self, args):

        self.config = {
            "episode_len": args["episode_len"],
            "state_dim": args["state_dim"],
            "chunk_size": args["chunk_size"],
            "camera_names": args["camera_names"]
        }

        self.args = args
        print(f"evaluation configs: {self.args}")
        self._load_model_config()

        self.observation_window = None
        self.policy = None
        self.prev_instr = None
        self.lang_embeddings = None

        self._make_poliy()

    def _load_model_config(self):
        model_config_path = self.args["model_config_path"]
        print(f"model_config_path: {model_config_path}")
        with open(model_config_path, "r") as fp:
            self.model_config = yaml.safe_load(fp)

    # def _interpolate_action(self, curr_action):
    #     steps = np.concatenate(
    #         (
    #             np.array(self.args.arm_steps_length),
    #             np.array(self.args.arm_steps_length),
    #         ),
    #         axis=0,
    #     )
    #     diff = np.abs(curr_action - self.prev_action)
    #     step = np.ceil(diff / steps).astype(int)
    #     step = np.max(step)
    #     if step <= 1:
    #         return curr_action[np.newaxis, :]
    #     new_actions = np.linspace(self.prev_action, curr_action, step + 1)
    #     return new_actions[1:]

    def _make_poliy(self):
        print("loading rdt model from: ")
        print(self.args["pretrained_rdt_model_path"])
        self.policy = create_model(
            args=self.model_config,
            dtype=torch.bfloat16,
            pretrained=self.args["pretrained_rdt_model_path"],
            pretrained_text_encoder_name_or_path=self.args["pretrained_text_encoder_path"],
            pretrained_vision_encoder_name_or_path=self.args["pretrained_vision_model_path"],
            control_frequency=self.args["ctrl_freq"],
        )

    def process_obs(self, obs):
        if self.observation_window is None:
            self.observation_window = deque(maxlen=2)

            self.observation_window.append(
                {
                    "proprio": None,
                    "images": {
                        self.config["camera_names"][0]: None,
                        self.config["camera_names"][1]: None,
                        self.config["camera_names"][2]: None,
                    },
                }
            )

        rob_obs = obs["robot_obs"]
        rob_obs = torch.from_numpy(rob_obs).float().cuda()

        self.observation_window.append(
            {
                "proprio": rob_obs,
                "images": {
                    self.config["camera_names"][0]: obs["rgb_obs"]["rgb_static"],
                    self.config["camera_names"][1]: obs["rgb_obs"]["rgb_gripper"],
                    self.config["camera_names"][2]: None,
                },
            }
        )

    def _process_action(self, actions):
        # if self.args.use_actions_interpolation:
        #     processed_actions = self._interpolate_action(action)
        # else:
        # actions = np.squeeze(actions)
        # print(f"actions shape: {actions.shape}, {actions}")
        actions[..., -1] = np.where(actions[..., -1] > 0, 1, -1)
        # eef_position = actions[..., :3]
        # eef_euler = actions[..., 3:6]

        # return (eef_position, eef_euler, gripper_action)
        return actions

    def reset(self):
        self.observation_window = None
        self.policy.reset()

    def step(self, obs, goal):
        image_arrs = [
            self.observation_window[-2]["images"][self.config["camera_names"][0]],
            self.observation_window[-2]["images"][self.config["camera_names"][1]],
            self.observation_window[-2]["images"][self.config["camera_names"][2]],
            self.observation_window[-1]["images"][self.config["camera_names"][0]],
            self.observation_window[-1]["images"][self.config["camera_names"][1]],
            self.observation_window[-1]["images"][self.config["camera_names"][2]],
        ]

        # image = rearrange(obs_dict[k], "h w c -> c h w")
        images = [
            PImage.fromarray(arr) if arr is not None else None for arr in image_arrs
        ]
        proprio = self.observation_window[-1]["proprio"].unsqueeze(0)
        # print(f"proprio shape: {proprio.shape}")
        # proprio = np.expand_dims(self.observation_window[-1]["proprio"], axis=0)
        if goal != self.prev_instr:
            self.lang_embeddings = self.policy.encode_instruction(goal)

        # time_start = time.time()
        actions = (
            self.policy.step(
                proprio=proprio, images=images, text_embeds=self.lang_embeddings
            )
            .squeeze(0)
            .cpu()
            .numpy()
        )
        actions = self._process_action(actions)

        self.prev_instr = goal
        # print(f"[step] action: {actions[0]}")
        # print(f"Model inference time: {time.time() - time_start} s")

        return actions
