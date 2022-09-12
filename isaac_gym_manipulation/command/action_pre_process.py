from typing import Union, Any
import torch
from abc import ABC, abstractmethod
from isaac_gym_manipulation.isaac_utils.actor_state import ActorState
from isaac_gym_manipulation.envs.utils import ik_get_dof_target, compute_osc_torques
from isaacgym.torch_utils import tensor_clamp, to_torch
from isaac_gym_manipulation.envs.utils import Pair


class ActionProcessing(ABC):
    """
    Converts actions to control space. actions => (forces , PID position targets ... )
    """

    def init(self, actor_state: ActorState, dof_range: Pair) -> None:
        """
        dof_range : the range on which to action will be applied.
        """
        self.actor_state = actor_state
        self.dof_range = dof_range

    @abstractmethod
    def to_control_space(self, actions: torch.Tensor, dt: float) -> Any:
        pass

    def current_val_of_target(self) -> Any:
        return NotImplementedError()


class TorqueActProcessing(ActionProcessing):
    def __init__(self, power_scale: Union[torch.Tensor, int, float]) -> None:
        self.power_scale = power_scale

    def init(self, actor_state: ActorState, dof_range: Pair):
        super().init(actor_state, dof_range)
        self.moto_effort = self.actor_state.effort_limits[
            self.dof_range[0] : self.dof_range[1]
        ]

    def to_control_space(self, actions: torch.Tensor, dt: float) -> Any:
        efforts = actions * self.moto_effort * self.power_scale
        return efforts


class DOF_PositionActProcessing(ActionProcessing):
    def init(self, actor_state: ActorState, dof_range: Pair):
        super().init(actor_state, dof_range)

        self.lower_limits = self.actor_state.dof_lower_limits[
            self.dof_range[0] : self.dof_range[1]
        ]
        self.upper_limits = self.actor_state.dof_upper_limits[
            self.dof_range[0] : self.dof_range[1]
        ]

    def to_control_space(self, actions: torch.Tensor, dt: float) -> Any:
        # TODO: Normalize DOF ?
        target_dof = tensor_clamp(
            actions,
            self.lower_limits,
            self.upper_limits,
        )
        return target_dof

    def current_val_of_target(self) -> Any:
        return self.actor_state.dof_pos[:, self.dof_range[0] : self.dof_range[1]]


class DOF_TranslationActProcessing(ActionProcessing):
    def __init__(
        self,
        action_scale: Union[torch.Tensor, int, float],
        dof_speed_scale: Union[torch.Tensor, int, float],
    ) -> None:
        self.action_scale = action_scale
        self.dof_speed_scale = dof_speed_scale

    def init(self, actor_state: ActorState, dof_range: Pair):
        super().init(actor_state, dof_range)
        self.current_dof_targets = actor_state.dof_pos

        self.lower_limits = self.actor_state.dof_lower_limits[
            self.dof_range[0] : self.dof_range[1]
        ]
        self.upper_limits = self.actor_state.dof_upper_limits[
            self.dof_range[0] : self.dof_range[1]
        ]

    def to_control_space(self, actions: torch.Tensor, dt: float) -> Any:
        current_dof = self.current_dof_targets
        target_dof = (
            current_dof + self.dof_speed_scale * dt * actions * self.action_scale
        )
        target_dof = tensor_clamp(
            actions,
            self.lower_limits,
            self.upper_limits,
        )
        self.current_dof_targets = target_dof
        return target_dof

    def current_val_of_target(self) -> Any:
        return self.actor_state.dof_pos[:, self.dof_range[0] : self.dof_range[1]]


class IK_ActProcessing(ActionProcessing):
    """
    /!\ WIP, controller not yet satisfactory
    """

    def __init__(
        self,
        target_rb: str,
        action_scale: Union[torch.Tensor, int, float] = 1,
        move_type: str = "translate_target",
        damping=0.05,
    ) -> None:
        self.damping = damping
        self.target_rb = target_rb
        self.action_scale = action_scale
        self.move_type = move_type

    def init(self, actor_state: ActorState, dof_range):
        super().init(actor_state, dof_range)
        self.num_envs = self.actor_state.num_envs
        self.actor_state.register_jacobian(self.target_rb, dof_range, id(self))
        self.target_rb_id = actor_state._rb_link_dict[self.target_rb]
        self.target_pos = self.actor_state.rb_pos[:, self.target_rb_id]

        self.lower_limits = self.actor_state.dof_lower_limits[
            self.dof_range[0] : self.dof_range[1]
        ]
        self.upper_limits = self.actor_state.dof_upper_limits[
            self.dof_range[0] : self.dof_range[1]
        ]

    def to_control_space(self, actions, dt) -> Any:
        device = actions.device
        # 1 Find the current position of the end effector :
        end_effector_pos = self.actor_state.rb_pos[:, self.target_rb_id]
        end_effector_rot = self.actor_state.rb_rot[:, self.target_rb_id]

        # 2 Compute the target position of the end effector :
        if self.move_type == "world_pos":
            end_effector_goal_pos = actions
        if self.move_type == "translate":
            end_effector_goal_pos = end_effector_pos + dt * actions * self.action_scale
        elif self.move_type == "translate_target":
            self.target_pos = self.target_pos + dt * actions * self.action_scale
            end_effector_goal_pos = self.target_pos
        end_effector_goal_rot = torch.tensor([0, 0, 1.0, 0], device=device).repeat(
            (self.num_envs, 1)
        )

        # 3 retrieve jacobian
        jacobian_target = self.actor_state.get_jacobian(id(self))

        # 4 execute ik algo to retrieve dof target position

        dof_translation = ik_get_dof_target(
            end_effector_pos,
            end_effector_rot,
            end_effector_goal_pos,
            end_effector_goal_rot,
            damping=self.damping,
            j_eef=jacobian_target,
            num_envs=self.num_envs,
            device=device,
        )
        dof_range = self.dof_range
        dof_targets = (
            self.actor_state.dof_pos[:, dof_range[0] : dof_range[1]] + dof_translation
        )
        # 5 clamp DOF to authorized positions :
        dof_targets = tensor_clamp(
            dof_targets,
            self.lower_limits,
            self.upper_limits,
        )

        # save values for debug :
        self.target_pos = end_effector_goal_pos
        self.current_pos = end_effector_pos
        return dof_targets

    def current_val_of_target(self) -> Any:
        return self.actor_state.rb_pos[:, self.target_rb_id]


class OSC_ActProcessing(ActionProcessing):
    """
    /!\ WIP, controller not yet satisfactory

    Use a PD controller to relate the force to apply to end effector
    based on the error of position between the target end effector
    and the current end effector.
    The target for the end effector is specified as a translation from the current position.

    To compute the force to apply to each DOF, the force is computed from the error of position
    And than the force is translated from the end effector space to join space.
    """

    def __init__(
        self,
        target_rb: str,
        action_scale=1,
        default_pos=None,
    ) -> None:
        self.target_rb = target_rb
        self.action_scale = action_scale

        self.default_dof_pos = default_pos

    def init(self, actor_state: ActorState, dof_range):
        super().init(actor_state, dof_range)
        self.device = self.actor_state.device
        self.num_envs = self.actor_state.num_envs
        self.actor_state.register_jacobian(self.target_rb, dof_range, id(self))
        self.actor_state.register_mass_matrix(dof_range, id(self))
        self.target_rb_id = actor_state._rb_link_dict[self.target_rb]
        self.effort_limits = self.actor_state.effort_limits[
            self.dof_range[0] : self.dof_range[1]
        ]

        self.lower_limits = self.actor_state.dof_lower_limits[
            self.dof_range[0] : self.dof_range[1]
        ]
        self.upper_limits = self.actor_state.dof_upper_limits[
            self.dof_range[0] : self.dof_range[1]
        ]
        num_dof = dof_range[1] - dof_range[0]
        if self.default_dof_pos == None:
            # check this default pos is valid ?
            self.default_dof_pos = (self.lower_limits + self.upper_limits) * 0.5
        self.default_dof_pos = to_torch(self.default_dof_pos, device=self.device)
        self.kp = to_torch([150.0] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.0] * num_dof, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        self.cmd_limit = to_torch(
            [0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device
        ).unsqueeze(0)
        self.dof_pos = self.actor_state.dof_pos[
            :, self.dof_range[0] : self.dof_range[1]
        ]
        self.dof_vel = self.actor_state.dof_vel[
            :, self.dof_range[0] : self.dof_range[1]
        ]

    def to_control_space(self, actions, dt) -> Any:
        device = actions.device
        # 1 Find the current position of the end effector :
        # end_effector_pos = self.actor_state.rb_pos[:, self.target_rb_id]
        # end_effector_rot = self.actor_state.rb_rot[:, self.target_rb_id]
        end_effector_vel = self.actor_state.rb_vel[:, self.target_rb_id]

        # 3 retrieve jacobian and mass matrix  :
        jac = self.actor_state.get_jacobian(id(self))
        mm = self.actor_state.get_mass_matrix(id(self))

        # 4 execute ik algo to retrieve dof target position
        actions = actions * self.cmd_limit / self.action_scale
        torques = compute_osc_torques(
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            mm=mm,
            dpose=actions,
            default_dof_pos=self.default_dof_pos,
            j_eef=jac,
            eef_vel=end_effector_vel,
            kp=self.kp,
            kd=self.kd,
            kd_null=self.kd_null,
            kp_null=self.kp_null,
            effort_limit=self.effort_limits,
            device=device,
        )
        # save values for debug :
        self.target_pos = (
            self.actor_state.rb_pos[:, self.target_rb_id, :] + actions[:, :3]
        )
        self.current_pos = self.actor_state.rb_pos[:, self.target_rb_id]
        return torques

    def current_val_of_target(self) -> Any:
        return self.actor_state.rb_pos[:, self.target_rb_id]
