import torch

from isaac_gym_manipulation.isaac_utils.actor_state import ActorState
from isaacgym.torch_utils import quat_rotate

# Helper code to identify if an object was grasped
# Right now, isaacgym can't identify if two object are colliding with GPU platform
# https://forums.developer.nvidia.com/t/get-contact-buffer-for-batch-simulation/191210
# https://forums.developer.nvidia.com/t/how-can-i-do-collision-detection-with-issac-gym/189308/2


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
        self.grap_detectors = {}

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

    def is_grapsed(self) -> bool:
        grasped = torch.logical_and(
            self.no_table_contact(),
            self.gripper_contact(),
            # self.gripper_is_closed(),
            # self.gripper_was_open_when_object_on_table(),
            # self.object_was_on_table_when_gripper_closed(target_actor),
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

    def gripper_is_closed(self):
        raise NotImplementedError()

    def gripper_was_open_when_object_on_table(self):
        raise NotImplementedError()

    def object_was_on_table_when_gripper_closed():
        raise NotImplementedError()
