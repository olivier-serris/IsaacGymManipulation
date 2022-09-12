import isaacgym  # importing isaac_gym before torch is mandatory.
import os
from isaac_gym_manipulation.envs.table_grasp_env import TableGraspEnv
import hydra
from omegaconf import OmegaConf
from isaacgym.torch_utils import to_torch
import numpy as np


@hydra.main(config_path=f"{os.getcwd()}/configs/", config_name="main_config.yaml")
def main_loop(cfg):

    cfg = OmegaConf.to_container(cfg, resolve=True)
    grasp_env = TableGraspEnv(
        cfg, sim_device="cpu", graphics_device_id=0, headless=False
    )
    grasp_env.reset()

    while True:
        actions = np.array(
            [grasp_env.act_space.sample() for _ in range(grasp_env.num_envs)]
        )
        actions = to_torch(actions, device=grasp_env.rl_device)
        observation, reward, done, info = grasp_env.step(actions)


if __name__ == "__main__":
    main_loop()
