from typing import Dict
import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import *


class IsaacTensorState:
    """
    Isaac Helper class.
    Contains most of the tensor API elements from isaacgym.
    """

    def __init__(self, gym, sim, device="cuda") -> None:
        self.sim = sim
        self.gym = gym
        self.rb_count = 0
        self.dof_count = 0
        self.actor_count = 0
        self.jacobian = {}
        self.mass_matrix = {}
        self._device = device

    def _register_tensor_views(self, envs):
        """
        Create tensor references that will store isaac physic states.
        """
        sim, gym = self.sim, self.gym
        self.num_dofs = gym.get_sim_dof_count(sim) // self.num_envs

        # state of the root rigidboy (position,orientation,linear velocity,angluar velocity), size[num_actor,13]:
        actor_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
        # state of all degree of freedom (position and velocity), size(num_dof,2):
        dof_state_tensor = gym.acquire_dof_state_tensor(sim)
        # state of all rigibodies (position,orientation,linear velocity,angluar velocity), size (num_bodies,13):
        rigidbody_tensor = gym.acquire_rigid_body_state_tensor(sim)

        # refresh the values of state tensors :
        success = gym.refresh_actor_root_state_tensor(sim)
        assert success, "failed refresh_actor_root_state_tensor"
        success = gym.refresh_dof_state_tensor(sim)
        assert success, "failed refresh_dof_state_tensor"
        success = gym.refresh_rigid_body_state_tensor(sim)
        assert success, "failed refresh_rigid_body_state_tensor"

        # views for the root state. Size [num_env,num_actor,13]
        self.root_state = gymtorch.wrap_tensor(actor_root_state_tensor).view(
            self.num_envs, -1, 13
        )
        # Views for the DOF states. Size [num_env,num_actor,2]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(
            self.num_envs, -1, 2
        )
        self.dof_pos = self.dof_state[:, :, 0]
        self.dof_vel = self.dof_state[:, :, 1]

        # Views for the rigidbody states. Size [num_env,num_rb,13]
        self.rb_state = gymtorch.wrap_tensor(rigidbody_tensor).view(
            self.num_envs, -1, 13
        )
        self.rigidbody_pos = self.rb_state[..., 0:3]
        self.rigidbody_rot = self.rb_state[..., 3:7]

        self.dof_position_target = torch.clone(self.dof_pos)
        self.dof_velocity_target = torch.zeros_like(self.dof_pos)

        self._n_dof = self.gym.get_env_dof_count(envs[0])

    def refresh_all(self):
        success = self.gym.refresh_actor_root_state_tensor(self.sim)
        assert success
        success = self.gym.refresh_dof_state_tensor(self.sim)
        assert success
        success = self.gym.refresh_rigid_body_state_tensor(self.sim)
        assert success
        # if len(self.jacobian) > 0:
        success = self.gym.refresh_jacobian_tensors(self.sim)
        assert success
        # if len(self.mass_matrix) > 0:
        success = self.gym.refresh_mass_matrix_tensors(self.sim)
        assert success

    def get_state(self, ids):
        root_state = self.root_state[ids, :]
        dof_state = self.dof_state[ids, :]
        return {"root_state": root_state, "dof_state": dof_state}

    def set_state(self, state, env_ids):
        dof_state = state["dof_state"]
        root_state = state["root_state"]

        # Reset actor root positions
        self.root_state[env_ids] = root_state[env_ids]
        success = self.gym.set_actor_root_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
        )
        assert success

        # Reset actor dof positions and velocity
        self.dof_state[env_ids] = dof_state[env_ids]
        success = self.gym.set_dof_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
        )
        assert success

        # Reset PID position targets :
        self.dof_position_target[env_ids] = self.dof_state[env_ids, :, 0]
        success = self.gym.set_dof_position_target_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_position_target),
        )
        assert success

        # reset PID velocity targets :
        self.dof_velocity_target[env_ids] = 0
        success = self.gym.set_dof_position_target_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_velocity_target),
        )
        assert success

    def register_jacobian(self, actor_name):
        jacobian = self.gym.acquire_jacobian_tensor(self.sim, actor_name)
        self.jacobian[actor_name] = gymtorch.wrap_tensor(jacobian)

    def register_mass_matrix(self, actor_name):
        mass_matrix = self.gym.acquire_mass_matrix_tensor(self.sim, actor_name)
        self.mass_matrix[actor_name] = gymtorch.wrap_tensor(mass_matrix)

    @property
    def num_envs(self):
        return self.gym.get_env_count(self.sim)

    @property
    def n_dof(self):
        return self._n_dof

    @property
    def device(self):
        return self._device
