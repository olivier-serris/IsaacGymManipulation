from typing import Dict
from isaac_gym_manipulation.envs.robot_solo_env import RobotSoloEnv
from isaac_gym_manipulation.envs.table_grasp_env import TableGraspEnv

task_map: Dict[str, RobotSoloEnv] = {
    "RobotSolo": RobotSoloEnv,
    "TableGrasp": TableGraspEnv,
}
