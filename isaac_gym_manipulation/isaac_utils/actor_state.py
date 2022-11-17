from typing import Any, Dict, List
import torch
from isaacgym.torch_utils import *
from bidict import bidict
from isaac_gym_manipulation.envs.utils import Pair
from isaac_gym_manipulation.isaac_utils.isaac_tensor_state import IsaacTensorState


class View:
    def __setattr__(self, __name: str, __value: Any) -> None:
        """Any modification made to tensor attribute of this class is made in place :

        # first setting :
        myView.my_tensor = dof_tensor[5:10]

        When myView.my_tensor is modified with :
        myView.my_tensor = new_tensor       # <- does not modify original_tensor values
        it will be automatically replaced by:
        myView.my_tensor[:] = new_tensor    # <- Also modify original_tensor values

        """
        if __name in self.__dict__:
            self.__dict__[__name][:] = __value
        else:
            super().__setattr__(__name, __value)


class DOF_View(View):
    def __init__(self, dof_state: torch.Tensor, dof_id_min, dof_id_max) -> None:
        self.state = dof_state[:, dof_id_min:dof_id_max, :]
        self.pos = self.state[..., 0]  # shape (n_env,n_dof)
        self.vel = self.state[..., 1]  # shape (n_env,n_dof)


class RB_View(View):
    def __init__(self, rb_state: torch.Tensor, rb_id_min, rb_id_max) -> None:
        self.state = rb_state[:, rb_id_min:rb_id_max, :]
        self.pos = self.state[..., 0:3]  # shape (n_env,n_rb,3)
        self.rot = self.state[..., 3:7]  # shape (n_env,n_rb,4)
        self.vel = self.state[..., 7:]  # shape (n_env,n_rb,3)


class RootView(View):
    def __init__(self, root_state: torch.Tensor, actor_index) -> None:
        self.state = root_state[:, actor_index, :]
        self.pos = self.state[..., 0:3]  # shape (n_env,3)
        self.rot = self.state[..., 3:7]  # shape (n_env,4)
        self.vel = self.state[..., 7:]  # shape (n_env,3)


class ActorState:
    """
    Allow to access the physic states of an actor for all environements.
    Includes root position, DOF and rigibody.
    """

    def __init__(
        self, name: str, asset, ranges: Dict[str, Pair], tensorState: IsaacTensorState
    ) -> None:
        self.name = name
        self.asset = asset
        self.tensorState: IsaacTensorState = tensorState
        self._actor_sim_ids = []

        # construct slices that will be updated when the isaac refresh functions are called
        self.dof_id_min = ranges["dof"][0]
        self.dof_id_max = ranges["dof"][1]
        self.rb_id_min = ranges["rb"][0]
        self.rb_id_max = ranges["rb"][1]
        self.actor_index = ranges["actor_index"]

        self._dof_dict = bidict(tensorState.gym.get_asset_dof_dict(asset))
        self._rb_link_dict = bidict(tensorState.gym.get_asset_rigid_body_dict(asset))

        self.num_dofs = tensorState.gym.get_asset_dof_count(asset)

        props = tensorState.gym.get_asset_dof_properties(asset)
        self._dof_lower_limits, self._dof_upper_limits = [], []
        self._effort_limit = []
        for i in range(self.num_dofs):
            if not props["hasLimits"][i]:
                props["effort"][i] = 0
                print(
                    f"WARNING : missing default efforts limits values in urdf for [{i}] {self._dof_dict.inverse[i]}"
                )
            self._dof_lower_limits.append(props["lower"][i])
            self._dof_upper_limits.append(props["upper"][i])
            self._effort_limit.append(props["effort"][i])

        self.props = props
        self._dof_lower_limits = to_torch(
            self._dof_lower_limits, device=tensorState.device
        )
        self._dof_upper_limits = to_torch(
            self._dof_upper_limits, device=tensorState.device
        )
        self._effort_limit = to_torch(self._effort_limit, device=tensorState.device)

        self.jacobian = None
        self.mass_matrix = None

        self._infos = {}

    def _register_views(self, tensorState: IsaacTensorState):
        self._dof_view = self.dof_view(tensorState.dof_state)
        self._rb_view = self.rb_view(tensorState.rb_state)
        self._root_view = self.root_view(tensorState.root_state)

    def add_sim_id(self, sim_id):
        self._actor_sim_ids.append(sim_id)

    def register_jacobian(self, target_rb: str, ik_tree_range: Pair, key: Any):
        self.tensorState.register_jacobian(self.name)
        if self.jacobian is None:
            self.jacobian = {}
        jac = self.tensorState.jacobian[self.name]
        target_id = self._rb_link_dict[target_rb]
        self.jacobian[key] = jac[:, target_id, :, ik_tree_range[0] : ik_tree_range[1]]

    def register_mass_matrix(self, ik_tree_range: Pair, key: Any):
        self.tensorState.register_mass_matrix(self.name)
        if self.mass_matrix is None:
            self.mass_matrix = {}
        mm = self.tensorState.mass_matrix[self.name]
        self.mass_matrix[key] = mm[
            :, ik_tree_range[0] : ik_tree_range[1], ik_tree_range[0] : ik_tree_range[1]
        ]

    def get_jacobian(self, key: Any):
        return self.jacobian[key]

    def get_mass_matrix(self, key: Any):
        return self.mass_matrix[key]

    def dof_view(self, dof_state: torch.Tensor) -> DOF_View:
        return DOF_View(dof_state, self.rb_id_min, self.rb_id_max)

    def rb_view(self, rb_state: torch.Tensor) -> RB_View:
        return RB_View(rb_state, self.rb_id_min, self.rb_id_max)

    def root_view(self, root_state: torch.Tensor) -> RootView:
        return RootView(root_state, self.actor_index)

    # Asset properties

    @property
    def effort_limits(self):
        return self._effort_limit

    @property
    def dof_lower_limits(self):
        return self._dof_lower_limits

    @property
    def dof_upper_limits(self):
        return self._dof_upper_limits

    @property
    def dof_dict(self):
        return self._dof_dict

    @property
    def rb_dict(self):
        return self._rb_link_dict

    @property
    def num_envs(self):
        return self.tensorState.num_envs

    @property
    def device(self):
        return self.tensorState.device

    @property
    def infos(self):
        return self._infos

    @property
    def actor_sim_ids(self):
        return self._actor_sim_ids

    # Tensor state properties :*

    @property
    def dof(self) -> DOF_View:
        return self._dof_view

    @property
    def rb(self) -> RB_View:
        return self._rb_view

    @property
    def root(self) -> RootView:
        return self._root_view
