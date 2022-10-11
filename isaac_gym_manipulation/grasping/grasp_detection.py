from typing import Dict
import torch

from isaac_gym_manipulation.isaac_utils.actor_state import ActorState
from isaac_gym_manipulation.envs.utils import batched_dot
from isaacgym.torch_utils import quat_rotate

# Helper code to identify if an object was grasped
# Right now, isaacgym can't identify if two object are colliding with GPU platform
# https://forums.developer.nvidia.com/t/get-contact-buffer-for-batch-simulation/191210
# https://forums.developer.nvidia.com/t/how-can-i-do-collision-detection-with-issac-gym/189308/2

STABLE: int = 5
POS_MOVED_THRESHOLD: float = 0.001
DEGREE_MOVED_THRESHOLD: float = 1


class GraspManager:
    def __init__(
        self,
        table_height: int,
        lift_height: int = 0.25,
        dist_to_gripper: int = 0.25,
    ) -> None:
        self.table_height = table_height
        self.lift_height = lift_height
        self.dist_to_gripper = dist_to_gripper
        self.grap_detectors: Dict[int, GraspDetector] = {}

    def track_grasped(
        self, target_id, target_actor: ActorState, gripper_pos: torch.Tensor
    ):
        assert target_id not in self.grap_detectors
        self.grap_detectors[target_id] = GraspDetector(
            target_actor,
            gripper_pos,
            self.table_height,
            self.lift_height,
            self.dist_to_gripper,
        )

    def is_grapsed(self, target_id):
        return self.grap_detectors[target_id].is_grapsed()

    def gripper_contact(self, target_id):
        return self.grap_detectors[target_id].gripper_contact()

    def no_table_contact(self, target_id):
        return self.grap_detectors[target_id].no_table_contact()

    def first_gripper_contact(self, target_id, timestep):
        return self.grap_detectors[target_id].first_gripper_contact(timestep)


class GraspDetector:
    def __init__(
        self,
        target_actor: ActorState,
        gripper_pos: torch.Tensor,
        table_height: int,
        lift_height: int = 0.25,
        gripper_radius: int = 0.2,
    ) -> None:
        self.target_actor = target_actor
        self.table_height = table_height
        self.gripper_pos = gripper_pos
        self.lift_height = lift_height
        self.gripper_radius = gripper_radius
        self.start_pos = torch.clone(target_actor.root.pos)
        self.start_rot = torch.clone(target_actor.root.rot)
        self.device = target_actor.device
        self.has_moved = torch.zeros(
            target_actor.root.pos.shape[0], device=self.device, dtype=bool
        )

    def is_grapsed(self) -> bool:
        grasped = torch.logical_and(
            self.no_table_contact(),
            self.gripper_contact(),
        )
        return grasped

    # Grasp Elements :
    def no_table_contact(self):
        """
        Can't access collision information with isaac gpu,
        for now we only check if th object is above the table.
        For now we just verify that the center of the object is
        lift_height cm above the table.
        """
        # retrieve the position of the center of the object
        actor = self.target_actor
        repeated_centroid = actor.infos["centroid"].repeat(actor.root.rot.shape[0], 1)
        target_pos = quat_rotate(actor.root.rot, repeated_centroid) + actor.root.pos

        table_contact = (
            target_pos[:, 2] - actor.infos["radius"]
            > self.table_height + self.lift_height
        )
        return table_contact

    def gripper_contact(self):
        """
        Can't access collision information with isaac gpu,
        For now we just verify check a sphere-sphere collision
        with a sphere centered on the gripper and a sphere centered on the obejct.
        """
        # retrieve the position of the center of the object
        actor = self.target_actor
        repeated_centroid = actor.infos["centroid"].repeat(actor.root.rot.shape[0], 1)
        target_pos = quat_rotate(actor.root.rot, repeated_centroid) + actor.root.pos
        # x^2 is faster than sqrt(x)
        dist_to_gripper = torch.sum((target_pos - self.gripper_pos) ** 2, dim=1)
        gripper_contact = (
            dist_to_gripper < (self.gripper_radius + actor.infos["radius"]) ** 2
        )

        return gripper_contact

    def first_gripper_contact(self, timestep: torch.Tensor):
        root = self.target_actor.root

        # Store the start position & rotation of the target object at the STABLE=5 timstep
        self.start_pos = torch.where(
            timestep.expand(root.pos.shape[::-1]).T == STABLE,
            torch.clone(root.pos),
            self.start_pos,
        )
        self.start_rot = torch.where(
            timestep.expand(root.rot.shape[::-1]).T == STABLE,
            torch.clone(root.rot).detach(),
            self.start_rot,
        )
        # euclidian (squared) distance between start pos anc current pos
        dist_to_start = torch.sum((root.pos - self.start_pos) ** 2, dim=1)
        # angle difference between 2 quaternions :
        rot_distance = torch.acos(2 * batched_dot(root.rot, self.start_rot) ** 2 - 1)

        # the object has moved either if the position is more than a threshold or
        # if the angle has changed more than a degree threshold.
        has_moved = torch.logical_or(
            dist_to_start > POS_MOVED_THRESHOLD**2,
            rot_distance > DEGREE_MOVED_THRESHOLD / 180 * torch.pi,
        )
        self.has_moved = torch.logical_or(has_moved, self.has_moved)

        # the object is considered not moved before the STABLE timestep :
        self.has_moved = torch.where(
            timestep > STABLE,
            self.has_moved,
            torch.zeros_like(self.has_moved, dtype=bool),
        )
        return self.has_moved
