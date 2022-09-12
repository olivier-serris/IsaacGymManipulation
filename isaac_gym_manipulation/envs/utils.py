import typing
from isaacgym.torch_utils import *
import numpy as np
import torch


Pair = typing.Tuple[int, int]


def get_rand_dof_pos(dof_properties):
    lower_limits = dof_properties["lower"]
    upper_limits = dof_properties["upper"]
    lower_limits = np.where(
        lower_limits == np.finfo(lower_limits.dtype).min, 0, lower_limits
    )
    upper_limits = np.where(
        upper_limits == np.finfo(upper_limits.dtype).max, 1, upper_limits
    )
    ranges = upper_limits - lower_limits

    num_dofs = len(ranges)

    pos_targets = lower_limits + ranges * np.random.random(num_dofs).astype("f")
    return pos_targets


def print_dof(gym, asset, dof_id):
    dof_names = gym.get_asset_dof_names(asset)
    dof_type = gym.get_asset_dof_type(asset, dof_id)
    props = gym.get_asset_dof_properties(asset)
    dof_config_str = (
        f"[{dof_id}]{dof_names[dof_id]}:\n"
        + f"Type:{gym.get_dof_type_string(dof_type)}\n"
        + f"limits ({props['lower'][dof_id]}:.2f,{props['upper'][dof_id]})\n"
        + f"stiffness: {props['stiffness'][dof_id]} damping: {props['damping'][dof_id]}"
    )
    print(dof_config_str)


#### Inverse kinematics #####


def ik_get_dof_target(
    current_pos, current_rot, goal_pos, goal_rot, damping, j_eef, num_envs, device
):
    """
    current_pos : current position of the effector we want to control
    tensor(num_env,3)
    current_rot : current rotation of the effector we want to control
    tensor(num_env,3)
    goal_pos : target position of the effector we want to control
    tensor(num_env,"?,?")
    goal_rot : target rotation of the effector we want to control
    tensor(num_env,?,?)

    dpose : the different between current pos and target pos
    damping : damping factor of the damped least square method
    j_eef : jacobian entries corresponding to hand body

    """
    # TODO : can reduce the number of required args (ex : num_envs)
    pos_err = goal_pos - current_pos
    orn_err = orientation_error(goal_rot, current_rot)
    dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
    pos_action = control_ik(dpose, damping, j_eef, num_envs, device)
    return pos_action


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def control_ik(dpose, damping, j_eef, num_envs, device):
    """
    dpose : the different between current pos and target pos
    damping : damping factor of the damped least square method
    j_eef : jacobian entries corresponding to hand body
    """
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping**2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
    return u


class Orn_Uhlen:
    def __init__(self, n_actions, mu=0, theta=0.15, sigma=0.2):
        self.n_actions = n_actions
        self.X = np.ones(n_actions) * mu
        self.mu = mu
        self.sigma = sigma
        self.theta = theta

    def reset(self):
        self.X = np.ones(self.n_actions) * self.mu

    def sample(self):
        dX = self.theta * (self.mu - self.X)
        dX += self.sigma * np.random.randn(self.n_actions)
        self.X += dX
        return torch.FloatTensor(self.X)


#### OSC :


def compute_osc_torques(
    dof_pos,
    dof_vel,
    mm,
    dpose,
    default_dof_pos,
    j_eef,
    eef_vel,
    kp,
    kd,
    kp_null,
    kd_null,
    effort_limit,
    device,
):
    """
    dof_pos : the positions of all the DOF used to control the movment of the end effecctor
    dof_vel : the velocity of the DOF used to control the movmen of the end effector
    mm : mass matrix
    dpose : the tranlation execute (num_env,6) 3 components for the translation in position, and 3 components for the rotation.
    default_dof_pos : ?
    j_eef : the jacobian of the end effector position with respect to the controllable joints
    eef_vel : the veloicty of the position of the end effector
    kp : coefficient of the PD controller od the OSC algo
    kp : coefficient of the PD controller od the OSC algo
    kp_null : ?
    kd_null : ?
    effort_limit : the maximum authorized value for the effort of each dof
    """
    q, qd = dof_pos, dof_vel
    num_dof = dof_pos.shape[-1]

    # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
    # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/

    mm_inv = torch.inverse(mm)
    m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    m_eef = torch.inverse(m_eef_inv)

    # Transform our cartesian action `dpose` into joint torques `u`
    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (kp * dpose - kd * eef_vel).unsqueeze(-1)

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    j_eef_inv = m_eef @ j_eef @ mm_inv
    u_null = kd_null * -qd + kp_null * (
        (default_dof_pos - q + np.pi) % (2 * np.pi) - np.pi
    )
    # u_null[:, 7:] *= 0 # ,je ne comprends pas.
    u_null = mm @ u_null.unsqueeze(-1)
    u += (
        torch.eye(num_dof, device=device).unsqueeze(0)
        - torch.transpose(j_eef, 1, 2) @ j_eef_inv
    ) @ u_null

    # Clip the values to be within valid effort range
    u = u.squeeze(-1)
    u = tensor_clamp(
        u,
        -effort_limit.unsqueeze(0),
        effort_limit.unsqueeze(0),
    )
    return u
