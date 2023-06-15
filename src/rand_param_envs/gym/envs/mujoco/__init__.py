from src.rand_param_envs.gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from src.rand_param_envs.gym.envs.mujoco.ant import AntEnv
from src.rand_param_envs.gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from src.rand_param_envs.gym.envs.mujoco.hopper import HopperEnv
from src.rand_param_envs.gym.envs.mujoco.walker2d import Walker2dEnv
from src.rand_param_envs.gym.envs.mujoco.humanoid import HumanoidEnv
from src.rand_param_envs.gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from src.rand_param_envs.gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from src.rand_param_envs.gym.envs.mujoco.reacher import ReacherEnv
from src.rand_param_envs.gym.envs.mujoco.swimmer import SwimmerEnv
from src.rand_param_envs.gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
