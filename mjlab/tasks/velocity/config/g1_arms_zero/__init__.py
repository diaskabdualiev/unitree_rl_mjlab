from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import unitree_g1_arms_zero_env_cfg
from .rl_cfg import unitree_g1_arms_zero_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-ArmsZero",
  env_cfg=unitree_g1_arms_zero_env_cfg(),
  play_env_cfg=unitree_g1_arms_zero_env_cfg(play=True),
  rl_cfg=unitree_g1_arms_zero_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
