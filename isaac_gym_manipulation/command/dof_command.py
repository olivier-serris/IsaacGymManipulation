from dataclasses import dataclass, field
import typing
from isaacgym import gymapi
from enum import Enum
import torch
import yaml
from isaac_gym_manipulation.isaac_utils.isaac_tensor_manager import IsaacTensorManager
from isaac_gym_manipulation.isaac_utils.isaac_tensor_state import IsaacTensorState
from isaac_gym_manipulation.isaac_utils.actor_state import ActorState
from isaac_gym_manipulation.command.action_pre_process import ActionProcessing
from isaacgym import gymutil, gymtorch, gymapi
from isaac_gym_manipulation.envs.utils import Pair
import numpy as np


class ControllerType(Enum):
    effort = 1
    dof_abs_pos = 2  # The actions are interpreted as DOF values that are directly set within isaacgym
    dof_pid_pos = 3  # The actions are interpreted as PIDs position-targets for each DOF
    dof_pid_vel = 4  # The actions are interpreted as PIDs velocity-targets for each DOF


# Maps a controller type to a isaacgym drivemode.
# Specifying a drivemode for a specific DOF enables isaac instructions (apply a force / set a pid target ... )
ACTION_TO_DRIVE_MODE = {
    ControllerType.effort: gymapi.DOF_MODE_EFFORT,
    ControllerType.dof_abs_pos: gymapi.DOF_MODE_POS,
    ControllerType.dof_pid_pos: gymapi.DOF_MODE_POS,
    ControllerType.dof_pid_vel: gymapi.DOF_MODE_VEL,
}


@dataclass
class DOF_Command:
    """
    A DOF_Commands allow to control a set of DOF for a specific actor.
    Control types are specified in the ControllerType enum.

    This class is reponsible for transforming the action of the agent to
    low level isaacgym instructions.
    """

    target_actor: str  # key of the target Actor to control
    dof_range: Pair  # IDs controlled DOF  in the actor
    controller_str: str  # string version of the ControllerType enum : specify the control space used
    controller_type: ControllerType = field(init=False)
    action_range: Pair  # corresponding entries in the actor action
    action_pre_process: ActionProcessing  # specify class responsible for the conversion Action -> Control Space
    pd_values: typing.Union[list, str, None] = None

    def __post_init__(self):
        self.controller_type = ControllerType[self.controller_str]
        if isinstance(self.pd_values, str):
            self.pd_values = yaml.safe_load(open(self.pd_values))

    def setup_props(self, dof_dict, props) -> None:
        ids = list(range(*self.dof_range))
        for dof_id in ids:
            props["driveMode"][dof_id] = self.drive_mode
            if self.drive_mode in [gymapi.DOF_MODE_POS, gymapi.DOF_MODE_VEL]:
                dof_name = self.pd_values[dof_id]
                props["stiffness"][dof_id] = self.pd_values[dof_name]["p"]
                props["damping"][dof_id] = self.pd_values[dof_name]["d"]
            else:
                props["stiffness"][dof_id] = 0
                props["damping"][dof_id] = 0

    def init(self, actor_state: ActorState):
        self.action_pre_process.init(actor_state, self.dof_range)
        actor_state = self.action_pre_process.actor_state
        self.world_dof_min = actor_state.dof_id_min + self.dof_range[0]
        self.world_dof_max = actor_state.dof_id_min + self.dof_range[1]
        assert (
            actor_state.dof_id_min
            <= self.world_dof_min
            <= self.world_dof_max
            <= actor_state.dof_id_max
        )

    def to_control_space(self, actions, dt):
        """
        Maps the raw action given the agent to the control space.
        """
        return self.action_pre_process.to_control_space(actions, dt)

    def current_val_of_target(self):
        return self.action_pre_process.current_val_of_target()

    def update_global_dof_pos(self, dof_state, command_val):
        """ """
        dof_state[:, self.world_dof_min : self.world_dof_max, :0] = command_val
        return dof_state

    def update_global_dof_val(self, dof_val, command_val):
        dof_val[:, self.world_dof_min : self.world_dof_max] = command_val
        return dof_val

    @property
    def drive_mode(self):
        return ACTION_TO_DRIVE_MODE[self.controller_type]


class CommandManager:
    """
    Contains a list of DOF_Command
    """

    def __init__(self, num_envs) -> None:
        self.command_dict: typing.Dict[ControllerType, DOF_Command] = {}
        self.num_envs = num_envs

    def init(self, isaac: IsaacTensorManager):
        """Must be called on after all actors have been created"""
        for command in self.iterate():
            actor_state = isaac.actor_dict[command.target_actor]
            command.init(actor_state)

    def add(self, command: DOF_Command):
        if command.controller_type not in self.command_dict:
            self.command_dict[command.controller_type] = []
        self.command_dict[command.controller_type].append(command)

    def iterate(self) -> typing.Generator[DOF_Command, None, None]:
        for commands in self.command_dict.values():
            for command in commands:
                yield command

    def get_command_for(self, actor_name, dof_id):
        """
        Returns the command respeonsible for controlling a specific DOF of an actor
        """
        for command in self.iterate():
            if (
                command.target_actor == actor_name
                and command.dof_range[0] <= dof_id < command.dof_range[1]
            ):
                return command
        return None

    def get_commands_of_type(
        self, controller_type: ControllerType
    ) -> typing.List[DOF_Command]:
        """
        Returns a list containing all the commands of the specified type.
        """
        return self.command_dict.get(controller_type, [])

    def execute_commands(self, isaac_state: IsaacTensorState, actions, dt):
        """
        Convert the action to isaac low level instructions:
        For each kind of instructions (Effort, PID ...) , executes these steps :
            1/ extract the entries of an action relevant for a specific command
            2/ convert the action to a control space command(ex : Effort)
            3/ Combine all commands for a specific type of control
            4/ transmit the combined command to isaac _gym.
        """
        gym = isaac_state.gym
        sim = isaac_state.sim

        # TODO : Optimization might be to re-use tensor instead of recreating them.
        # # EFFORT control :
        commands = self.get_commands_of_type(ControllerType.effort)
        if commands:
            dof_efforts = torch.zeros(
                self.num_envs, isaac_state.n_dof, device=isaac_state.device
            )
            for command in commands:
                act = actions[:, command.action_range[0] : command.action_range[1]]
                control_act = command.to_control_space(act, dt)
                dof_efforts = command.update_global_dof_val(dof_efforts, control_act)
            success = gym.set_dof_actuation_force_tensor(
                sim, gymtorch.unwrap_tensor(dof_efforts)
            )
            assert success

        # Absolute pos control :
        commands = self.get_commands_of_type(ControllerType.dof_abs_pos)
        if commands:
            # TODO : Optimization might not need to clone.
            dof_state = torch.clone(isaac_state.dof_state)
            for command in commands:
                act = actions[:, command.action_range[0] : command.action_range[1]]
                control_act = command.to_control_space(act, dt)
                dof_state = command.update_global_dof_pos(dof_state, control_act)
            success = gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))
            assert success

        # PID target pos control :
        commands = self.get_commands_of_type(ControllerType.dof_pid_pos)
        if commands:
            pos_target = torch.zeros(
                self.num_envs, isaac_state.n_dof, device=self.device
            )
            pos_target = isaac_state.dof_position_target
            for command in commands:
                act = actions[:, command.action_range[0] : command.action_range[1]]
                control_act = command.to_control_space(act, dt)
                pos_target = command.update_global_dof_val(pos_target, control_act)
            success = gym.set_dof_position_target_tensor(
                sim, gymtorch.unwrap_tensor(pos_target)
            )
            assert success

    def sanity_checks(self):
        def range_overlap(range_pairs):
            elements = set()
            overlap = set()
            for pair in range_pairs:
                new_elements = set(range(*pair))
                overlap = overlap.union(set.intersection(elements, new_elements))
                elements = elements.union(new_elements)
            return elements, overlap

        def is_contiguous(list):
            array = np.array(list)
            return (np.diff(array) == 1).all()

        # check DOF indexes
        range_pairs = [command.dof_range for command in self.iterate()]
        _, overlap = range_overlap(range_pairs)
        if len(overlap) > 0:
            raise Exception(f"The DOF {overlap} are controlled by multiples commands")

        # check action indexes
        range_pairs = [command.action_range for command in self.iterate()]
        action_indexes, overlap = range_overlap(range_pairs)
        if len(overlap) > 0:
            raise Exception(f"The action {overlap} controls multiple DOF")

        action_indexes = sorted(action_indexes)
        if not is_contiguous(action_indexes):
            raise Exception("some of the action indexes are not used")
        else:
            self.n_actions = len(action_indexes)
