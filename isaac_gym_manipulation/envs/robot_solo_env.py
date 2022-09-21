import copy
import numpy as np
import torch

from isaacgym import gymapi
from isaacgym.torch_utils import *

import hydra
from bidict import bidict

from .base.vec_task import VecTask
from isaac_gym_manipulation.debugger.debuggers import DefaultDebugger
from isaac_gym_manipulation.isaac_utils.isaac_tensor_manager import IsaacTensorManager
from isaac_gym_manipulation.command.dof_command import CommandManager
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver(
    "resolve_default", lambda default, arg: default if arg == "" else arg
)


class RobotSoloEnv(VecTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):

        self.cfg = cfg
        self.cfg["sim"]["up_axis"] = "z"
        self.cfg["sim"]["gravity"] = [0.0, 0.0, -9.81]
        self.up_axis = self.cfg["sim"]["up_axis"]
        self.up_axis_idx = 2
        self.dt = cfg["sim"]["dt"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.debug_mode = not headless
        self.scene_cfg = cfg["env"]["scene"]
        self.agent_name = self.scene_cfg["agent"]["name"]
        self.reset_noise = self.cfg["env"]["reset_noise"]
        self.asset_root = self.scene_cfg["asset_dir"]

        ### create commands based on config:
        self.command_manager = CommandManager(cfg["env"]["numEnvs"])
        for actor_name in cfg["command"]:
            for name, command_args in cfg["command"][actor_name].items():
                command = hydra.utils.instantiate(command_args)
                self.command_manager.add(command)
        self.command_manager.sanity_checks()
        if "numActions" not in cfg["env"]:
            cfg["env"]["numActions"] = self.command_manager.n_actions

        super().__init__(
            config=self.cfg,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
        )
        ### Create tensor helper class to access actor states :
        self.isaac_tensor_manager.register_tensor_views(self.envs)
        self.isaac_tensor_manager.refresh_all()

        # Save the current state of the simulation for reset
        self.start_state = self.isaac_tensor_manager.get_state(
            torch.arange(self.num_envs)
        )
        start_dof = self.scene_cfg["agent"].get("start_dof", 0)
        self.start_state["dof_state"][..., 0] = to_torch(start_dof, device=self.device)
        self.start_state["dof_state"][..., 1] = 0  # the speed of all DOF is set to 0
        self.command_manager.init(self.isaac_tensor_manager)
        self.isaac_tensor_manager.refresh_all()

        # reset all environments on startup :
        self.reset_idx(
            torch.arange(self.num_envs, device=self.device), with_noise=self.reset_noise
        )

        # visualisation and debug options:
        self.setup_camera()
        if self.debug_mode:
            self.set_up_debugger()

    def get_state(self, env_ids: torch.Tensor = None) -> dict:
        """Returns a dictionary containing the state of the simulation"""
        if env_ids == None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        return self.isaac_tensor_manager.get_state(env_ids)

    def set_state(self, state: dict, env_ids: torch.Tensor = None) -> None:
        if env_ids == None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.isaac_tensor_manager.set_state(state, env_ids)

    def setup_camera(self) -> None:
        cam_pos = gymapi.Vec3(0, -2, 3)
        cam_target = gymapi.Vec3(0, 2, 0)
        self.gym.viewer_camera_look_at(self.viewer, self.envs[0], cam_pos, cam_target)

    def set_up_debugger(self, debugger=None) -> None:
        if debugger == None:
            self.env_debugger = DefaultDebugger(self)
        else:
            self.env_debugger = debugger

    def get_main_agent_asset(self):
        """
        Save properties of the tiago asset and returns the tiago asset.
        The properties saved contains information about the drive mode od DOFs
        based on the commands specified in the cfg.
        """
        agent_cfg = self.scene_cfg["agent"]
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = agent_cfg["flip_visual_attachments"]
        asset_options.disable_gravity = not agent_cfg["gravity"]
        asset_options.fix_base_link = agent_cfg["fix_base"]

        # asset_options.collapse_fixed_joints = (
        #     True  #      Merge links that are connected by fixed joints.
        # )
        urdf_path = agent_cfg["urdf_path"]
        agent_asset = self.gym.load_asset(
            self.sim, self.asset_root, urdf_path, asset_options
        )
        props = self.gym.get_asset_dof_properties(agent_asset)
        num_dof = self.gym.get_asset_dof_count(agent_asset)
        dof_dict = bidict(self.gym.get_asset_dof_dict(agent_asset))

        # set_up props with commands :
        for command in self.command_manager.iterate():
            command.setup_props(dof_dict, props)  # in-place modification of props.

        # setup uncontrolled props :
        for i in range(num_dof):
            command = self.command_manager.get_command_for(self.agent_name, i)
            if command is None:
                props["stiffness"][i] = 0
                props["damping"][i] = 0
                props["driveMode"][i] = gymapi.DOF_MODE_NONE

        # self.agent_dof_props = props
        return agent_asset, props

    def create_sim(self):
        """Create all parallel environments"""
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self.isaac_tensor_manager = IsaacTensorManager(self.gym, self.sim, self.device)
        self._create_envs(self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))
        self._create_ground_plane()

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, spacing, num_per_row) -> None:
        # load robots props:

        agent_asset, agent_dof_props = self.get_main_agent_asset()
        agent_start_tr = gymapi.Transform()
        agent_start_tr.p = gymapi.Vec3(*self.scene_cfg["agent"]["pos"])
        agent_start_tr.r = gymapi.Quat.from_euler_zyx(0, 0, 0)

        # Create environments :
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.envs = []
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            agent_actor = self.isaac_tensor_manager.create_actor(
                env,
                asset=agent_asset,
                transform=agent_start_tr,
                name=self.scene_cfg["agent"]["name"],
                collision_group=i,
            )
            self.gym.set_actor_dof_properties(env, agent_actor, agent_dof_props)
            self.envs.append(env)

    def compute_reward(self, actions: torch.Tensor) -> None:
        """Computes the reward and reset buffer"""
        self.rew_buf[:] = torch.zeros(self.num_envs).to(self.device)
        self.reset_buf[:] = self.progress_buf >= self.max_episode_length - 1

    def compute_observations(self) -> torch.Tensor:
        """Computes the reward and reset buffer"""
        return torch.zeros(self.num_observations, device=self.device)

    def reset_idx(self, env_ids: torch.Tensor, with_noise=True):
        """Reset the selected environments"""
        agent_state = self.isaac_tensor_manager.actor_dict[self.agent_name]
        num_agent_dof = agent_state.num_dofs
        reset_state = copy.deepcopy(self.start_state)
        if with_noise:
            # take a view of the reset state with only actor dof
            dof_view = agent_state.dof_view(reset_state["dof_state"])
            # add noise to actor DOF
            dof_view.pos += 0.25 * (
                torch.rand((len(env_ids), num_agent_dof), device=self.device) - 0.5
            )
            dof_view.pos = tensor_clamp(
                dof_view.pos,
                agent_state.dof_lower_limits,
                agent_state.dof_upper_limits,
            )
        self.isaac_tensor_manager.set_state(reset_state, env_ids)
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def reset(self, with_noise=True):
        self.reset_idx(
            torch.arange(self.num_envs, device=self.device), with_noise=with_noise
        )
        return super().reset()

    def pre_physics_step(self, actions: torch.Tensor):
        """Converts actions to isaac instructions"""
        assert actions.shape == (
            self.num_envs,
            self.num_actions,
        ), f"action tensor has shape {actions.shape}, which should be {(self.num_envs,self.num_actions)} =>(num_env,num_action)"
        self.actions = actions.clone().to(self.device)
        self.command_manager.execute_commands(
            self.isaac_tensor_manager.tensor_state, self.actions, self.dt
        )

    def post_physics_step(self):
        self.progress_buf += 1

        # auto reset all environments that need to be reset.
        if (self.reset_buf > 0).any():
            env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            self.reset_idx(env_ids, with_noise=self.reset_noise)
        self.isaac_tensor_manager.refresh_all()
        self.compute_observations()
        self.compute_reward(self.actions)

        if self.debug_mode and self.env_debugger is not None:
            self.env_debugger.input_events()
            self.env_debugger.draw()

    # Getters :
    def get_props(self, actor_name: str):
        """
        Returns the asset properties of selected actor
        """
        return self.isaac_tensor_manager.actor_dict[actor_name].props

    def get_sim_time(self):
        return self.gym.get_sim_time(self.sim)
