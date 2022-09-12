
### About this repository
This repository contains RL environments for the NVIDIA Isaac Gym high performance environments based on [Isaac Gym Env](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs).

### Installation 

Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then follow the installation instructions in the documentation.  
Once Isaac Gym is installed and samples work within your current python environment, install this repo:
```
pip install -e .
```
### Creating an environement : 
```
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

```
## Reach
... WIP ... 
## Table Grasp 
#### Observations : 
#### Rewards :
#### Reset : 

Like other isaacgymenvs, autoreset are used.  
Each of the DOF of the agent is set to it's starting state then adds uniform noise of magnitude 0.25 to all DOF.  
( /!\ Nothing prevents self-collisions from happening). 

## Configs files

The environments takes in a config file to specify the parameters of the simulation.
It's a dictionnary so any method can be used to construct it. 

This repository uses hydra with the following hierarchy : 
```
configs
|   ycb_table_env_config.yaml
└───command # Various methods to control the agent
|---|   franka_dof.yaml
|---|   ...
└───scene # robot + other objects in the scene
|---|   franka_ycb.yaml
|---|   ...
```
The ycb_table_env_config file indicates which commands and which scenes are loaded.  
The way the config works are through the use of config groups. You can learn more about how these [here](https://hydra.cc/docs/tutorials/structured_config/config_groups/).

### Debug tools : 
Each environements can execute a Debugger class.  
This class handles visuals gizmos and shortcuts commands.  
The default debugger class proposes the folowing shortcuts : 

* R : Reset all environements
* 1-9 : Selected an index $i$
* F5 : Save the current environement state to the $i^{th}$ save slot 
* F9 : Load the current environment state from the $i^{th}$ save slot
* P : Prints the actor states, first DOFs, then rigidbody positions.

## ISAAC GYM Helper Classes 

Environment provided in the repository are based on:  
https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.

<!-- Each classes inherit the VecTask. https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/base/vec_task.py -->

### Access to physic state

<!-- Isaac gym gives access to physic state with tensors containing every objects of every environements : 
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states) # this tensor have shape (n_env, n_rigibody, 13)
0
You can then access specific DOF by manually saving the id of the rigidbody you care about.  

Added helpers :  -->

When actors are created with IsaacTensorManager.create_actor(), you can access to the actor state directly : 
```
actor_state = isaac_tensor_manager.actor_dict('my_actor')
actor_state.root.pos    # shape : (n_env, 3)
actor_state.root.rot    # shape : (n_env, 4)
actor_state.dof.pos     # shape : (n_env, n_actor_dof)
actor_state.dof.vel     # shape : (n_env, n_actor_dof)
actor_state.rb.pos      # shape : (n_env, n_actor_rb, 3)
actor_state.rb.rot      # shape : (n_env, n_actor_rb, 4)
actor_state.rb.vel      # shape : (n_env, n_actor_rb, 3)
```
### Commands 

The DOF_command class allows to specify a list of DOF ids and the method to control them. 
