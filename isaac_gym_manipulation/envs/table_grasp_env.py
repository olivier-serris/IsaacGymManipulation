from xmlrpc.client import Boolean
from isaac_gym_manipulation.debugger.debuggers import GraspDebugger
from isaac_gym_manipulation.envs.robot_solo_env import RobotSoloEnv
from isaacgym import gymapi
import numpy as np
import torch
from isaac_gym_manipulation.grasping.grasp_detection import GraspManager
from isaac_gym_manipulation.grasping.utils import get_ycb_mesh_path_from_urdf
import trimesh
from isaacgym.torch_utils import to_torch


class TableGraspEnv(RobotSoloEnv):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        super().__init__(cfg, sim_device, graphics_device_id, headless)

        table_height = self.scene_cfg["table"]["pos"][2]

        # Find the rb of the agent gripper.
        agent_actor = self.isaac_tensor_manager.actor_dict[self.agent_name]
        gripper_rb_id = agent_actor._rb_link_dict[
            self.scene_cfg["agent"]["gripper_rb_key"]
        ]
        gripper_rb_pos = agent_actor.rb.pos[:, gripper_rb_id, :]

        # register object to detect grasp informations.
        grasp_cfg = self.scene_cfg["agent"]["grasp"]
        self.grasp_manager = GraspManager(
            table_height=table_height,
            lift_height=grasp_cfg["lift_height"],
            dist_to_gripper=grasp_cfg["dist_to_gripper"],
        )
        for object_id in self.object_ids:
            obj_actor = self.isaac_tensor_manager.actor_dict[object_id]
            self.grasp_manager.track_grasped(
                target_id=object_id, target_actor=obj_actor, gripper_pos=gripper_rb_pos
            )
        self.nb_objects = len(self.object_ids)

    def set_up_debugger(self, debugger=None):
        grasp_cfg = self.scene_cfg["agent"]["grasp"]
        if debugger == None:
            self.env_debugger = GraspDebugger(
                self,
                self.isaac_tensor_manager.actor_dict[self.agent_name],
                self.scene_cfg["agent"]["gripper_rb_key"],
                grasp_cfg["dist_to_gripper"],
            )
        else:
            debugger = debugger

    def _create_envs(self, spacing, num_per_row) -> None:
        """
        This function create each parallelized environment and the actors inside it.
        This function must re-written entirely for each modification :
            All actors must be added to an env before creating the next env.
            Adding actors to envs out of order can result in errors.
        """
        # load robots props:
        asset_root = self.scene_cfg["asset_dir"]
        agent_asset, agent_dof_props = self.get_main_agent_asset()
        agent_start_tr = gymapi.Transform()
        agent_start_tr.p = gymapi.Vec3(*self.scene_cfg["agent"]["pos"])
        agent_start_tr.r = gymapi.Quat.from_euler_zyx(0, 0, 0)
        n_agent_bodies = self.gym.get_asset_rigid_body_count(agent_asset)
        n_agent_shapes = self.gym.get_asset_rigid_shape_count(agent_asset)

        # table props :
        table_pos = gymapi.Vec3(*self.scene_cfg["table"]["pos"])
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_dimensions = (
            np.array([1.2, 1.2, table_thickness]) * self.scene_cfg["table"]["scale"]
        )
        table_asset = self.gym.create_box(self.sim, *table_dimensions, table_opts)
        n_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        n_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)

        # Define start pose for table
        table_tr = gymapi.Transform()
        table_tr.p = table_pos  # gymapi.Vec3(*table_pos)
        table_tr.r = agent_start_tr.r

        self._table_surface_pos = table_pos + gymapi.Vec3(*[0, 0, table_thickness / 2])

        # load objects assets :
        asset_options = gymapi.AssetOptions()
        asset_options.convex_decomposition_from_submeshes = True  # with this parameters, it seems that collision from vhacd file are used.
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        # overrides values for COM (ycb urdf asset values are wrong) : .
        asset_options.override_inertia = True
        asset_options.override_com = True
        # placeholder :
        cube_asset = self.gym.create_box(self.sim, *([0.1, 0.1, 0.1]))
        # asset_options.use_mesh_materials = True
        object_assets = {}
        n_objects_bodies = 0
        n_objects_shapes = 0
        for obj_key, object_config in self.scene_cfg["YCB_objects"].items():
            object_asset = self.gym.load_asset(
                self.sim, asset_root, object_config["urdf_path"], asset_options
            )
            n_objects_bodies += self.gym.get_asset_rigid_body_count(object_asset)
            n_objects_shapes += self.gym.get_asset_rigid_shape_count(object_asset)
            object_assets[obj_key] = object_asset

        # Create environments :
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.envs = []
        self.object_ids = []
        n_body_per_env = n_agent_bodies + n_table_bodies + n_objects_bodies
        n_shapes_per_env = n_agent_shapes + n_table_shapes + n_objects_shapes

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env)

            # aggregation is an optimization mecanism described in IsaacGym_Preview_4_Package/isaacgym/docs/programming/physics.html?highlight=aggregate#aggregates
            self.gym.begin_aggregate(env, n_body_per_env, n_shapes_per_env, True)

            # Create main actor :
            agent_actor = self.isaac_tensor_manager.create_actor(
                env,
                asset=agent_asset,
                transform=agent_start_tr,
                name=self.scene_cfg["agent"]["name"],
                collision_group=i,
            )
            self.gym.set_actor_dof_properties(env, agent_actor, agent_dof_props)

            # Create table
            self.isaac_tensor_manager.create_actor(
                env, table_asset, table_tr, "table", i
            )

            if "YCB_objects" not in self.scene_cfg:
                continue
            # Create objects on table
            for obj_key, object_config in self.scene_cfg["YCB_objects"].items():
                object_asset = object_assets[obj_key]
                pose = gymapi.Transform()
                pose.p = self._table_surface_pos + gymapi.Vec3(*object_config["pos"])
                pose.r = gymapi.Quat(*object_config.get("rot", (0, 0, 0, 1)))
                self.isaac_tensor_manager.create_actor(
                    env, object_asset, pose, obj_key, i
                )

                if obj_key not in self.object_ids:
                    mesh_path = get_ycb_mesh_path_from_urdf(
                        asset_root, object_config["urdf_path"]
                    )
                    mesh = trimesh.load(mesh_path)
                    infos = {
                        "centroid": to_torch(
                            mesh.bounding_box.centroid, device=self.device
                        ),
                        "radius": max(mesh.bounding_box.extents) / 2,
                    }
                    self.isaac_tensor_manager.actor_dict[obj_key].infos.update(infos)
                    self.object_ids.append(obj_key)
            self.gym.end_aggregate(env)

    def compute_reward(self, actions):
        """
        Returns 1 if at least one object is grasped.
        Need to be optimized and jitted.
        """
        self.gripper_collide_object = torch.zeros(self.num_envs, self.nb_objects).to(
            self.device
        )
        self.no_table_contact = torch.zeros(self.num_envs, self.nb_objects).to(
            self.device
        )
        grasped = torch.zeros(self.num_envs, self.nb_objects).to(self.device)

        for i, object_id in enumerate(self.object_ids):
            self.gripper_collide_object[:, i] = self.grasp_manager.gripper_contact(
                object_id
            )
            self.no_table_contact[:, i] = self.grasp_manager.no_table_contact(object_id)
            grasped[:, i] = self.grasp_manager.is_grapsed(object_id)

            #  in isaacgym ve_env the classic info dict is called extras :
            self.extras[
                f"object_{i}_first_gripper_contact"
            ] = self.grasp_manager.first_gripper_contact(object_id, self.progress_buf)
        rewards, _ = torch.max(grasped, dim=1)

        self.rew_buf = rewards
        self.reset_buf = self.progress_buf >= self.max_episode_length - 1

    def compute_observations(self):
        self.isaac_tensor_manager.refresh_all()
        robot = self.isaac_tensor_manager.actor_dict[self.agent_name]
        return robot.dof.pos
