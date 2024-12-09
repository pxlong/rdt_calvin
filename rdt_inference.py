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
            "episode_len": args.max_publish_step,
            "state_dim": args.state_dim,
            "chunk_size": args.chunk_size,
            "camera_names": args.camera_names,
        }

        self.args = args
        self._load_model_config()

        # self.prev_action = None
        self.observation_window = None
        self.lang_embeddings = None
        self.policy = None

        self._make_poliy()

    def _load_model_config(self):
        model_config_path = current_dir + self.args.model_config_path
        print(f"model_config_path: {model_config_path}")
        with open(model_config_path, "r") as fp:
            model_config = yaml.safe_load(fp)
        self.args.model_config = model_config

    def _interpolate_action(self, curr_action):
        steps = np.concatenate(
            (
                np.array(self.args.arm_steps_length),
                np.array(self.args.arm_steps_length),
            ),
            axis=0,
        )
        diff = np.abs(curr_action - self.prev_action)
        step = np.ceil(diff / steps).astype(int)
        step = np.max(step)
        if step <= 1:
            return curr_action[np.newaxis, :]
        new_actions = np.linspace(self.prev_action, curr_action, step + 1)
        return new_actions[1:]

    def _make_poliy(self):
        print(f"Loading RDT model from: {self.args.pretrained_rdt_model_path}")
        self.policy = create_model(
            args=self.args.model_config,
            dtype=torch.bfloat16,
            pretrained=self.args.pretrained_rdt_model_path,
            pretrained_text_encoder_name_or_path=self.args.pretrained_text_encoder_path,
            pretrained_vision_encoder_name_or_path=self.args.pretrained_vision_model_path,
            control_frequency=self.args.ctrl_freq,
        )

    def _process_obs(self, obs):
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
        # rob_obs = torch.from_numpy(rob_obs).float().cuda()

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

    def _process_action(self, action):
        if self.args.use_actions_interpolation:
            processed_actions = self._interpolate_action(action)
        else:
            processed_actions = action[np.newaxis, :]

        self.prev_action = action.copy()
        return processed_actions

    def reset(self):
        self.observation_window = None
        self.policy.reset()

    def step(self, obs, goal):
        self._process_obs(obs)

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
        proprio = self.observation_window[-1]["rob_obs"].unsqueeze(0)
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
        # print(f"Model inference time: {time.time() - time_start} s")

        return actions
