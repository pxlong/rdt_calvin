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
REPLACEMENTS = {
    "_": " ",
    "1f": " ",
    "4f": " ",
    "-": " ",
    "50": " ",
    "55": " ",
    "56": " ",
}

TASK_D_VALIDATION_TASKS = {
    'rotate_red_block_right': [13006, 16818, 36382] #, 71207, 71217, 82352, 8338, 8340, 16419, 16408], 
    # 'rotate_red_block_left': [8149, 8166, 8246, 31293, 37091, 37555, 37564, 50037, 49932, 53935], 
    # 'rotate_blue_block_right': [8530, 8457, 8544, 18826, 21403, 42714, 42702, 56811, 56889, 8453], 
    # 'rotate_blue_block_left': [18737, 18740, 36510, 36500, 62437, 62547, 81870, 81948, 86107, 85567], 
    # 'rotate_pink_block_right': [8035, 8020, 19789, 19936, 33267, 33254, 38811, 47632, 54533, 54529], 
    # 'rotate_pink_block_left': [3090, 34684, 36311, 43416, 43424, 47832, 63652, 82245, 90126, 90112], 
    # 'push_red_block_right': [88543, 97689, 2385, 2393, 88537, 97693, 97697, 88553, 88546, 2389], 
    # 'push_red_block_left': [37028, 69829, 69610, 88625, 97773, 97767, 69609, 97777, 37030, 69605], 
    # 'push_blue_block_right': [21743, 21612, 21616, 21732, 21736, 21739, 21744, 21617, 21740, 21735], 
    # 'push_blue_block_left': [40140, 94874, 40143, 56968], 
    # 'push_pink_block_right': [34371, 78955, 82742, 92956, 92970, 78951, 78960, 92974, 92968, 34379], 
    # 'push_pink_block_left': [20014, 20003, 34273, 34289, 64126, 93057, 20021, 20010, 64134, 64122], 
    # 'move_slider_left': [1911, 5744, 5756, 6745, 7123, 6766, 6770, 6751, 7110, 9000], 
    # 'move_slider_right': [1653, 1632, 1644, 3610, 3604, 4974, 4968, 4951, 9765, 9777], 
    # 'open_drawer': [85, 2098, 2108, 3913, 5106, 5096, 6320, 7472, 9881, 11080], 
    # 'close_drawer': [632, 2757, 2764, 4636, 4652, 6011, 9359, 10542, 10536, 10552], 
    # 'lift_red_block_table': [2819, 4380, 6369, 10794, 13731, 16804, 19229, 19219, 23655, 24877], 
    # 'lift_red_block_slider': [518, 10314, 10320, 11651, 11662, 12475, 17812, 29357, 29367, 37861], 
    # 'lift_red_block_drawer': [2288, 2300, 5272, 7548, 7559, 19000, 22031, 27208, 35231, 44343], 
    # 'lift_blue_block_table': [1023, 1012, 6497, 6502, 9198, 15466, 16257, 16264, 17585, 18805], 
    # 'lift_blue_block_slider': [298, 4123, 4136, 6082, 15763, 15771, 21201, 24681, 24691, 25771], 
    # 'lift_blue_block_drawer': [2186, 2174, 5183, 11129, 11137, 14735, 14743, 17905, 17915, 23323], 
    # 'lift_pink_block_table': [787, 3517, 3800, 3788, 8887, 24337, 24327, 25494, 28863, 38794], 
    # 'lift_pink_block_slider': [1367, 6645, 6631, 23856, 25909, 25913, 46251, 57235, 57225, 95248], 
    # 'lift_pink_block_drawer': [2659, 2680, 7777, 7794, 12821, 12816, 19387, 23502, 27317, 27307], 
    # 'place_in_slider': [2237, 2247, 3717, 3741, 3730, 5497, 5474, 5857, 5847, 9133], 
    # 'place_in_drawer': [579, 587, 595, 1295, 4409, 4208, 4199, 4428, 6419, 6531], 
    # 'stack_block': [3245, 3234, 3385, 3379, 3368, 5618, 8650, 8631, 13322, 13778], 
    # 'unstack_block': [3285, 3409, 3418, 3295, 5661, 8698, 8714, 13682, 13696, 14026], 
    # 'turn_on_lightbulb': [893, 879, 868, 1990, 2012, 4773, 6826, 10376, 11542, 13457], 
    # 'turn_off_lightbulb': [1519, 1527, 4013, 4004, 4023, 5931, 5910, 5926, 9446, 10715], 
    # 'turn_on_led': [4705, 4718, 6155, 6160, 6176, 7395, 11194, 11217, 11995, 12003], 
    # 'turn_off_led': [934, 938, 5013, 5009, 4996, 6590, 9543, 9559, 10451, 10470], 
    # 'push_into_drawer': [5365, 5356, 11331, 11402, 11355, 11393, 11335, 13221, 13217, 33794]
}


def clean_task_instruction(task_instruction: str, replacements: dict) -> str:
    """
    Clean up the natural language task instruction.
    """
    # Apply replacements to the string
    for old, new in replacements.items():
        task_instruction = task_instruction.replace(old, new)

    # Strip leading and trailing spaces
    cleaned_task_instruction = task_instruction.strip()

    return cleaned_task_instruction


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
    model, env, data_module, eval_dir=None, debug=False
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
    task_cfg = OmegaConf.load(Path(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    ))
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(
        conf_dir / "annotations/new_playtable_validation.yaml"
    )

    eval_dir = get_log_dir(eval_dir)
    # eval_sequences = get_sequences(num_seqs)
    # print(f"eval_sequences type: {type(eval_sequences)}")
    # print(f"eval_sequences: {eval_sequences}")
    dataset = data_module.val_dataloader().dataset.datasets["vis"]

    results = Counter()
    # if not debug:
    #     eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for task, ids in TASK_D_VALIDATION_TASKS.items():
        for i in ids:
            episode = dataset[int(i)]
            results[task] += rollout_episode(
                env, 
                model, 
                episode,
                task,
                task_oracle,
                val_annotations,
                eval_dir,
                i,
                debug
            )
        print(f"{task}: {results[task]} / {len(ids)}")

    print(f"SR: {sum(results.values()) / sum(len(x) for x in TASK_D_VALIDATION_TASKS.values()) * 100:.1f}%")
    return results



def rollout_episode(
    env, 
    model, 
    episode,
    task,
    task_oracle,
    val_annotations,
    eval_dir,
    task_idx,
    debug 
):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    reset_info = episode["state_info"]
    obs = env.reset(robot_obs=reset_info["robot_obs"][0], scene_obs=reset_info["scene_obs"][0])
    lang_annotation = val_annotations[task][0]

    instruction = clean_task_instruction(lang_annotation, REPLACEMENTS)
    print(f"task: {task}, lang_ann: {lang_annotation}, instr: {instruction}")
    if instruction != lang_annotation:
        print(f"*** instruction: {instruction}, lang_ann: {lang_annotation}")

    start_info = env.get_info()
    obs = env.get_obs()
    # get lang annotation for subtask
    model.reset()
    model.process_obs(obs)

    if debug:
        img_list = []

    action_buffer = []
    for step in range(EP_LEN):
        action_idx = step % model.config["chunk_size"]
        # print(f"action_idx: {action_idx}")
        if action_idx == 0:
            action_buffer = model.step(obs, instruction).copy()
            # print(f"rdt inferencing, action shape: {action_buffer.shape}")

        action = action_buffer[action_idx]
        # print(f"[evaluate] action.shape: {action.shape}, {action}")

        action = (action[..., :3], action[..., 3:6], np.array(action[..., 6]).reshape(1))
        # print(f"[evaluate] action: {action}")

        obs, _, _, current_info = env.step(action)
        model.process_obs(obs)

        if debug:
            img_copy = copy.deepcopy(obs["rgb_obs"]["rgb_static"])
            img_list.append(img_copy)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(
            start_info, current_info, {task}
        )
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"))
                clip = ImageSequenceClip(img_list, fps=30)
                clip.write_videofile(
                    os.path.join(eval_dir, f"{task}-{task_idx}-succ.mp4"),
                    codec="libx264", fps=30
                )
            return True

    if debug:
        print(colored("fail", "red"))
        clip = ImageSequenceClip(img_list, fps=30)
        clip.write_videofile(
            os.path.join(eval_dir, f"{task}-{task_idx}-fail.mp4"), 
            codec="libx264", fps=30
        )
    return False

    
def get_data_module(train_folder, dataset_path):
    train_cfg_path = Path(train_folder) / ".hydra/config.yaml"
    cfg = OmegaConf.load(train_cfg_path)
    # conf_path = f"{CALVIN_ROOT}/calvin_models/conf/datamodule/datasets"
    conf_path = "../calvin/calvin_models/conf/datamodule/datasets"
    print(f"*** conf_path: {conf_path}")

    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(conf_path)
        print("*** initialize hydra")
    datasets_cfg = hydra.compose("vision_lang.yaml", overrides=["lang_dataset.lang_folder=" + "lang_annotations"])
    print(f"*** datasets_cfg: {datasets_cfg}")
    print(f"*** datasets_path: {dataset_path}")
    cfg.datamodule.datasets = datasets_cfg
    cfg.datamodule.root_data_dir = dataset_path

    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    data_module.prepare_data()
    data_module.setup()

    return data_module

def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on multistep sequences with language goals."
    )
    parser.add_argument(
        "--train_folder",
        default="/mnt/petrelfs/longpinxin/data/calvin/model/D_D_static_rgb_baseline",
        type=str,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--dataset_path",
        default="/mnt/petrelfs/longpinxin/data/calvin/mini_task_D_D",
        type=str,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--calvin_checkpoint",
        type=str,
        default="/mnt/petrelfs/longpinxin/data/calvin/model/D_D_static_rgb_baseline/mcil_baseline.ckpt",
        help="Manually specify checkpoint path (default is latest). Only used for calvin_agent.",
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
    parser.add_argument(
        "--chunk_size",
        default=64,
        type=int,
        help="chunk_size to evaluate",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoint-1000",
        help="checkpoint to evaluate",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="default",
        help="tasks to evaluate",
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    # evaluate a custom model
    config["pretrained_rdt_model_path"] = os.path.join(
        config["pretrained_rdt_model_path"], args.ckpt)
    config["chunk_size"] = args.chunk_size
    model = RDTCalvinEvaluation(config)
    env = make_env(args.dataset_path)
    data_module = get_data_module(args.train_folder, args.dataset_path)

    # _, _, data_module = get_default_model_and_env(args.train_folder, args.dataset_path, args.calvin_checkpoint, env=None)

    default_eval_dir = "/mnt/petrelfs/longpinxin/ws/rdt_calvin/results"
    eval_dir = os.path.join(default_eval_dir, args.eval_dir) 
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir, exist_ok=True)
    # eval_sr_path = os.path.join(eval_dir, "success_rate.txt")
    # eval_result_path = os.path.join(eval_dir, "results.txt")

    # if args.task == "default":
    #     tasks = "new_playtable_tasks.yaml"
    # else:
    #     tasks = args.task + "_tasks.yaml"
    # print(f"tasks to evaluate: {tasks}")

    evaluate_policy(
        model, env, data_module, eval_dir, debug=args.debug
    )


if __name__ == "__main__":
    main()
