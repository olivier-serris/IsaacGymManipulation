from typing import Callable
from isaacgym import gymutil, gymtorch, gymapi
import math

import torch
from isaac_gym_manipulation.isaac_utils.actor_state import ActorState
import pickle
import os

QUICK_SAVE_FOLDER = "/tmp/quicksave/"


class DefaultDebugger:
    def __init__(self, env, *args, **kwargs) -> None:
        self.env = env
        self.gym = self.env.gym
        self.viewer = self.env.viewer
        self.events = {}
        self.state = None
        self.selected_index = 0

        self.suscribe_event(gymapi.KEY_R, "reset", self.on_reset)
        self.suscribe_event(gymapi.KEY_P, "print", self.on_print)
        self.suscribe_event(gymapi.KEY_1, "1", self.on_index)
        self.suscribe_event(gymapi.KEY_2, "2", self.on_index)
        self.suscribe_event(gymapi.KEY_3, "3", self.on_index)
        self.suscribe_event(gymapi.KEY_4, "4", self.on_index)
        self.suscribe_event(gymapi.KEY_5, "5", self.on_index)
        self.suscribe_event(gymapi.KEY_6, "6", self.on_index)
        self.suscribe_event(gymapi.KEY_7, "7", self.on_index)
        self.suscribe_event(gymapi.KEY_8, "8", self.on_index)
        self.suscribe_event(gymapi.KEY_9, "9", self.on_index)
        self.suscribe_event(gymapi.KEY_F5, "quick_save", self.on_quicksave)
        self.suscribe_event(gymapi.KEY_F9, "quick_load", self.on_quickload)

        if not os.path.exists(QUICK_SAVE_FOLDER):
            os.makedirs(QUICK_SAVE_FOLDER)

    def suscribe_event(self, key, name: str, on_activate: Callable):
        self.events[name] = on_activate
        self.gym.subscribe_viewer_keyboard_event(self.viewer, key, name)

    def draw(self):
        pass

    def input_events(self):
        for evt in self.gym.query_viewer_action_events(self.viewer):

            if evt.action in self.events.keys() and evt.value > 0:
                self.events[evt.action](evt)

    def on_reset(self, event):
        print("R pressed : Debugger FORCE RESET")
        self.env.reset(self.env.reset_noise)

    def on_print(self, event):
        print("P Pressed : All current DOF of env 0 : ")
        for key, actor in self.env.isaac_tensor_manager.actor_dict.items():
            print(key)
            print("DOF:")
            print(actor.dof.pos[0, :])
            print("RB_POS")
            print(actor.rb.pos[0, :])
            print("RB_ROT")
            print(actor.rb.rot[0, :])
            print("_" * 50)

    def on_quicksave(self, event):
        quick_save_full_path = os.path.join(
            QUICK_SAVE_FOLDER, f"save_{self.selected_index}.state"
        )
        print(f"quick save {self.selected_index}")
        self.state = self.env.get_state()
        with open(quick_save_full_path, "wb") as file:
            pickle.dump(self.state, file)

    def on_quickload(self, event):
        quick_save_full_path = os.path.join(
            QUICK_SAVE_FOLDER, f"save_{self.selected_index}.state"
        )

        try:
            with open(quick_save_full_path, "rb") as file:
                self.state = pickle.load(file)
            self.env.set_state(self.state)
            print(f"quick load {self.selected_index}")
        except FileNotFoundError:
            print(f"quick load {self.selected_index} FileNotFoundError")

    def on_index(self, event):
        self.selected_index = int(event.action)
        print(f"current_index : {self.selected_index}")


class GraspDebugger(DefaultDebugger):
    def __init__(
        self, env, actor_state, gripper_rb_name, dist_to_gripper, *args, **kwargs
    ) -> None:
        super().__init__(env, *args, **kwargs)
        self.target = None
        self.tensor_state = self.env.isaac_tensor_manager.tensor_state

        rb_id = actor_state._rb_link_dict[gripper_rb_name]
        self.gripper_pos = actor_state.rb.pos[:, rb_id, :]
        self.gripper_rot = actor_state.rb.rot[:, rb_id, :]

        self.yellow_sphere = gymutil.WireframeSphereGeometry(
            dist_to_gripper, 12, 12, gymapi.Transform(), color=(1, 1, 0)
        )
        self.box_geom = gymutil.WireframeBoxGeometry()
        self.suscribe_event(gymapi.KEY_T, "teleport", self.on_teleport_into_gripper)

        self.obj_spheres = {}
        self.obj_actor_dict = {}
        for obj_key in self.env.object_ids:
            actor = self.env.isaac_tensor_manager.actor_dict[obj_key]
            self.obj_actor_dict[obj_key] = actor
            self.obj_spheres[obj_key] = gymutil.WireframeSphereGeometry(
                actor.infos["radius"],
                12,
                12,
                gymapi.Transform(),
                color=(0, 0, 0),
            )

    def draw(self):

        gym = self.env.gym
        envs = self.env.envs
        gym.clear_lines(self.viewer)

        env_id = 0
        sphere_pose = gymapi.Transform()
        sphere_pose.p = gymapi.Vec3(*self.gripper_pos[env_id, :])
        sphere_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)
        gymutil.draw_lines(
            self.yellow_sphere,
            self.gym,
            self.viewer,
            envs[env_id],
            sphere_pose,
        )
        for obj_id, obj_key in enumerate(self.env.object_ids):
            actor = self.env.isaac_tensor_manager.actor_dict[obj_key]
            sphere_pose = gymapi.Transform()
            sphere_pose.r = gymapi.Quat(*actor.root.rot[env_id, :])
            sphere_pose.p = sphere_pose.r.rotate(
                gymapi.Vec3(*actor.infos["centroid"])
            ) + gymapi.Vec3(*actor.root.pos[env_id, :])
            color = torch.zeros(3)
            if self.env.no_table_contact[env_id, obj_id]:
                color += torch.tensor((0, 1, 0))
            if self.env.gripper_collide_object[env_id, obj_id]:
                color += torch.tensor((1, 0, 0))
            if self.env.extras[f"object_{obj_id}_first_gripper_contact"][env_id]:
                color += torch.tensor((0, 0, 1))
            gymutil.draw_lines(
                gymutil.WireframeSphereGeometry(
                    self.obj_actor_dict[obj_key].infos["radius"],
                    12,
                    12,
                    gymapi.Transform(),
                    color=tuple(color.tolist()),
                ),
                self.gym,
                self.viewer,
                envs[env_id],
                sphere_pose,
            )

    def on_teleport_into_gripper(self, event):
        print("teleport into gripper")
        if self.selected_index < len(self.env.object_ids):
            object_state: ActorState = self.env.isaac_tensor_manager.actor_dict[
                self.env.object_ids[self.selected_index]
            ]
            object_state.root.pos = self.gripper_pos
            object_state.root.rot = self.gripper_rot

            self.env.gym.set_actor_root_state_tensor(
                self.env.sim,
                gymtorch.unwrap_tensor(self.tensor_state.root_state),
            )
        else:
            print("no object numbered : ", self.selected_index)


class ReachDebugger(DefaultDebugger):
    """
    /!\ WIP
    """

    def __init__(self, env, *args, **kwargs) -> None:
        super().__init__(env, *args, **kwargs)
        self.target = None

        self.suscribe_event(gymapi.Ke)

    def update(self, current, target):
        self.current = current
        self.target = target
        sphere_pose = gymapi.Transform()
        self.sphere_geom = gymutil.WireframeSphereGeometry(
            0.02, 12, 12, sphere_pose, color=(1, 1, 0)
        )

    def draw(self):
        gym = self.env.gym
        envs = self.env.envs
        gym.clear_lines(self.viewer)
        for i in range(self.env.num_envs):
            if self.target is not None:
                sphere_pose = gymapi.Transform()
                sphere_pose.p = gymapi.Vec3(*self.target[i])
                sphere_pose.r = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)

                gymutil.draw_lines(
                    self.sphere_geom,
                    self.gym,
                    self.viewer,
                    envs[i],
                    sphere_pose,
                )
                target = self.target[i]
                current = self.current[i]
                self.gym.add_lines(
                    self.viewer,
                    envs[i],
                    1,
                    [
                        target[0],
                        target[1],
                        target[2],
                        current[0],
                        current[1],
                        current[2],
                    ],
                    [1, 0, 0],
                )
