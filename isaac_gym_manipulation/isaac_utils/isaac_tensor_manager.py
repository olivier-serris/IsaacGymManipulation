from typing import Dict
from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaac_gym_manipulation.isaac_utils.actor_state import ActorState, IsaacTensorState


class IsaacTensorManager:
    """
    Isaac Helper class.
    Used to register actor_states, gives safe and easy access to an actor physic state.
    """

    def __init__(self, gym, sim, device) -> None:
        self.sim = sim
        self.gym = gym
        self._actor_dict: Dict(str, ActorState) = {}
        self.rb_count = 0
        self.dof_count = 0
        self.actor_count = 0
        self.jacobian = {}
        self.mass_matrix = {}
        self.device = device
        self.tensor_state = IsaacTensorState(gym, sim, device)

    def refresh_all(self):
        self.tensor_state.refresh_all()

    def register_tensor_views(self, envs):
        """
        Establish the link with isaac gym to access the simulation data (ex : positoinn velocity ... )
        /!\ Can only be called once all the environements have been created.
        """
        self.tensor_state._register_tensor_views(envs)
        for actor in self.actor_dict.values():
            actor._register_views(self.tensor_state)

    def get_state(self, ids):
        return self.tensor_state.get_state(ids)

    def set_state(self, state, env_ids):
        self.tensor_state.set_state(state, env_ids)

    def create_actor(
        self,
        env,
        asset,
        transform,
        name,
        collision_group,
        collision_group_filter=-1,
        segmentation_id=0,
    ):
        actor_handle = self.gym.create_actor(
            env,
            asset,
            transform,
            name,
            collision_group,
            collision_group_filter,  # docs/programming/assets.html?highlight=collision#creating-actors
            segmentation_id,
        )
        if name not in self._actor_dict:
            self._actor_dict[name] = self._register_actor(name, asset)
        else:
            actor_index = self.gym.get_actor_index(env, actor_handle, gymapi.DOMAIN_SIM)
            self._actor_dict[name].add_sim_id(actor_index)
        return actor_handle

    def _register_actor(self, name, asset):
        n_dof = self.gym.get_asset_dof_count(asset)
        n_rb = self.gym.get_asset_rigid_body_count(asset)
        ranges = {
            "dof": (self.dof_count, self.dof_count + n_dof),
            "rb": (self.rb_count, self.rb_count + n_rb),
            "actor_index": self.actor_count,
        }
        self.rb_count += n_rb
        self.dof_count += n_dof
        self.actor_count += 1
        actor_state = ActorState(name, asset, ranges, self.tensor_state)
        return actor_state

    @property
    def actor_dict(self) -> Dict[str, ActorState]:
        return self._actor_dict

    @property
    def num_envs(self):
        return self.gym.get_env_count(self.sim)
