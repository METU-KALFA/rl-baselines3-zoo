from typing import Callable, Optional

import gymnasium as gym
from gymnasium.envs.registration import register

from rl_zoo3.wrappers import MaskVelocityWrapper

try:
    import pybullet_envs_gymnasium  # pytype: disable=import-error
except ImportError:
    pass

try:
    import highway_env  # pytype: disable=import-error
except ImportError:
    pass
else:
    # hotfix for highway_env
    import numpy as np

    np.float = np.float32  # type: ignore[attr-defined]

try:
    import custom_envs  # pytype: disable=import-error
except ImportError:
    pass

try:
    import gym_donkeycar  # pytype: disable=import-error
except ImportError:
    pass

try:
    import panda_gym  # pytype: disable=import-error
except ImportError:
    pass

try:
    import rocket_lander_gym  # pytype: disable=import-error
except ImportError:
    pass

try:
    import minigrid  # pytype: disable=import-error
except ImportError:
    pass

try:
    import assembly_learning  # pytype: disable=import-error
except ImportError:
    pass

# Register no vel envs
def create_no_vel_env(env_id: str) -> Callable[[Optional[str]], gym.Env]:
    def make_env(render_mode: Optional[str] = None) -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)
        env = MaskVelocityWrapper(env)
        return env

    return make_env


for env_id in MaskVelocityWrapper.velocity_indices.keys():
    name, version = env_id.split("-v")
    register(
        id=f"{name}NoVel-v{version}",
        entry_point=create_no_vel_env(env_id),  # type: ignore[arg-type]
    )

env_name = 'AssembleRL-{}-{}-v2'.format("ivar", "supervised")

register(
    id= env_name,
    entry_point='assembly_learning.envs:IvarCloudMujoco',
    kwargs={
        "furniture_name": "chair_ivar",
        "furniture_id": 4,
        "part_pos_dict": {"part0": [-0.5, 0.5, 0.3],
                    "part1": [0,  0.5, 0.3],
                    "part2": [0.5, -0.5, 0.05],
                    "part3": [0.2, -0.5, 0.05],
                    "part4": [-0.5, -0.5, 0.05]},
                    
        "camera_name": ["up", "front"],
        "camera_params": [[0, 0.25, 2, 1, 0, 0, 0, 480, 640], [0, -2.5, 1, 1, 0, 0, 1.2, 480, 640]],
        "light_params": [-0.5, -0.5, 1.5],
        "floor_params": [0, 0, 0, 10, 10, 1],
        "main_part": "part0",
        "nf_size": 32,
        "num_points": 256,
        "max_ep_length": 8,
        "points_threshold":0.015,
        "num_threshold": 100,
        "save_frames": False,
        "framerate": 120,
    },
)