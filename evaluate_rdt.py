import argparse
from collections import Counter, defaultdict
import logging
import os
import copy
from pathlib import Path
import sys
import json
import time
from moviepy.editor import ImageSequenceClip

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)

import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from calvin_env.envs.play_table_env import get_env
from rdt_inference import RDTCalvinEvaluation

logger = logging.getLogger(__name__)

CALVIN_ROOT = os.environ["CALVIN_ROOT"]

EP_LEN = 360
# NUM_SEQUENCES = 1000


def get_epoch(checkpoint):
    if "=" not in checkpoint.stem:
        return "0"
    checkpoint.stem.split("=")[1]


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


def evaluate_policy(
    model, env, num_seqs, eval_sr_path, eval_result_path, eval_dir=None, debug=False
):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    # conf_dir = Path(__file__).absolute().parents[2] / "conf"
    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(
        conf_dir / "annotations/new_playtable_validation.yaml"
    )

    eval_dir = get_log_dir(eval_dir)
    eval_sequences = get_sequences(num_seqs)
    # print(f"eval_sequences type: {type(eval_sequences)}")
    # print(f"eval_sequences: {eval_sequences}")

    results = []
    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    seq_i = 0
    for initial_state, eval_sequence in eval_sequences:
        print()
        print(f"\nSequence {seq_i + 1}/{num_seqs}")
        # print(f"initial_state: {initial_state}, eval_sequence: {eval_sequence}")
        result = evaluate_sequence(
            env,
            model,
            task_oracle,
            initial_state,
            eval_sequence,
            val_annotations,
            debug,
            eval_dir,
            seq_i,
        )
        results.append(result)

        if not debug:
            success_list = count_success(results)
            with open(eval_sr_path, "a") as f:
                line = f"{seq_i}/{num_seqs}: "
                for sr in success_list:
                    line += f"{sr:.3f} | "
                seq_i += 1
                line += "\n"
                f.write(line)

            eval_sequences.set_description(
                " ".join(
                    [
                        f"{i + 1}/5 : {v * 100:.1f}% |"
                        for i, v in enumerate(success_list)
                    ]
                )
                + "|"
            )
        else:
            seq_i += 1

    print_and_save(results, eval_sequences, eval_result_path, None)
    return results


def evaluate_sequence(
    env,
    model,
    task_checker,
    initial_state,
    eval_sequence,
    val_annotations,
    debug,
    eval_dir,
    seq_i,
):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0

    if debug:
        time.sleep(1)
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")

    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout(
            env,
            model,
            task_checker,
            subtask,
            val_annotations,
            debug,
            eval_dir,
            subtask_i,
            seq_i,
        )
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(
    env, model, task_oracle, subtask, val_annotations, debug, eval_dir, subtask_i, seq_i
):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ")
        time.sleep(0.5)

    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    print(f"lang_annotation: {lang_annotation}")
    model.reset()
    start_info = env.get_info()
    if debug:
        img_list = []

    for step in range(EP_LEN):
        # action = model.step(obs, lang_annotation)
        if step % model.config["chunk_size"] == 0:
            action_buffer = model.step(obs, lang_annotation).copy()
            print(f"rdt inferencing, step-{step}")

        # print(f"step: {step}")
        action = action_buffer[step % model.config["chunk_size"]]
        # print(f"[evaluate] action.shape: {action.shape}, {action}")

        action = (action[..., :3], action[..., 3:6], np.array(action[..., 6]).reshape(1))
        # print(f"[evaluate] action: {action}")

        obs, _, _, current_info = env.step(action)
        if debug:
            img_copy = copy.deepcopy(obs["rgb_obs"]["rgb_static"])
            img_list.append(img_copy)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(
            start_info, current_info, {subtask}
        )
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"))
                clip = ImageSequenceClip(img_list, fps=30)
                clip.write_videofile(
                    os.path.join(eval_dir, f"{seq_i}-{subtask_i}-{subtask}-succ.mp4"),
                    codec="libx264", fps=30
                )
            return True

    if debug:
        print(colored("fail", "red"))
        clip = ImageSequenceClip(img_list, fps=30)
        clip.write_videofile(
            os.path.join(eval_dir, f"{seq_i}-{subtask_i}-{subtask}-fail.mp4"), 
            codec="libx264", fps=30
        )
    return False


def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on multistep sequences with language goals."
    )
    parser.add_argument(
        "--dataset_path",
        default="/mnt/petrelfs/longpinxin/data/calvin",
        type=str,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/mnt/petrelfs/longpinxin/ws/rdt_calvin/configs/eval_configs.json",
        help="Path to the config file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info and visualize environment.",
    )
    parser.add_argument(
        "--eval_dir",
        default="run0",
        type=str,
        help="Where to log the evaluation results.",
    )
    parser.add_argument(
        "--num_seqs",
        default=10,
        type=int,
        help="num_sequences to evaluate",
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    # evaluate a custom model
    model = RDTCalvinEvaluation(config)
    env = make_env(args.dataset_path)

    default_eval_dir = "/mnt/petrelfs/longpinxin/ws/rdt_calvin/results"
    eval_dir = os.path.join(default_eval_dir, args.eval_dir) 
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir, exist_ok=True)
    eval_sr_path = os.path.join(eval_dir, "success_rate.txt")
    eval_result_path = os.path.join(eval_dir, "results.txt")

    evaluate_policy(
        model, env, args.num_seqs, 
        eval_sr_path, eval_result_path, eval_dir, 
        debug=args.debug
    )


if __name__ == "__main__":
    main()
