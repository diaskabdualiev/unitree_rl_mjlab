"""Unitree G1 velocity config with arms LOCKED at zero position.

Arms are completely excluded from the action space and held at zero
by the built-in position actuators. Only legs (12 DOF) + waist (3 DOF)
= 15 DOF are controlled by RL policy.
"""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise


# Joint name patterns for legs + waist only (15 DOF)
_LEGS_WAIST_JOINTS = (
  ".*_hip_pitch_joint",
  ".*_hip_roll_joint",
  ".*_hip_yaw_joint",
  ".*_knee_joint",
  ".*_ankle_pitch_joint",
  ".*_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
)


def unitree_g1_fixed_arms_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 flat terrain velocity with arms locked at zero."""
  cfg = make_velocity_env_cfg()

  # Get base robot config and set arm default positions to zero
  robot_cfg = get_g1_robot_cfg()
  robot_cfg.init_state.joint_pos.update({
    ".*_shoulder_pitch_joint": 0.0,
    ".*_shoulder_roll_joint": 0.0,
    ".*_shoulder_yaw_joint": 0.0,
    ".*_elbow_joint": 0.0,
    ".*_wrist_roll_joint": 0.0,
    ".*_wrist_pitch_joint": 0.0,
    ".*_wrist_yaw_joint": 0.0,
  })
  cfg.scene.entities = {"robot": robot_cfg}

  site_names = ("left_foot", "right_foot")

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

  # Flat terrain
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  # === ACTIONS: only legs + waist (15 DOF), arms excluded ===
  legs_waist_action_scale = {
    k: v for k, v in G1_ACTION_SCALE.items()
    if any(
      __import__("re").match(pat, k)
      for pat in (
        r".*_hip_pitch_joint", r".*_hip_roll_joint", r".*_hip_yaw_joint",
        r".*_knee_joint", r".*_ankle_pitch_joint", r".*_ankle_roll_joint",
        r"waist_yaw_joint", r"waist_roll_joint", r"waist_pitch_joint",
      )
    )
  }
  cfg.actions["joint_pos"] = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=_LEGS_WAIST_JOINTS,
    scale=legs_waist_action_scale,
    use_default_offset=True,
  )

  # === OBSERVATIONS: only legs + waist joints ===
  legs_waist_cfg = SceneEntityCfg("robot", joint_names=_LEGS_WAIST_JOINTS)

  cfg.observations["policy"] = ObservationGroupCfg(
    terms={
      "base_ang_vel": ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
        noise=Unoise(n_min=-0.2, n_max=0.2),
      ),
      "projected_gravity": ObservationTermCfg(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
      ),
      "command": ObservationTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "twist"},
      ),
      "joint_pos": ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": legs_waist_cfg},
        noise=Unoise(n_min=-0.01, n_max=0.01),
      ),
      "joint_vel": ObservationTermCfg(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": legs_waist_cfg},
        noise=Unoise(n_min=-1.5, n_max=1.5),
      ),
      "actions": ObservationTermCfg(func=mdp.last_action),
    },
    concatenate_terms=True,
    enable_corruption=True,
    history_length=1,
  )
  cfg.observations["critic"] = ObservationGroupCfg(
    terms={
      "base_lin_vel": ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_lin_vel"},
      ),
      "base_ang_vel": ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
      ),
      "projected_gravity": ObservationTermCfg(
        func=mdp.projected_gravity,
      ),
      "command": ObservationTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "twist"},
      ),
      "joint_pos": ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": legs_waist_cfg},
      ),
      "joint_vel": ObservationTermCfg(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": legs_waist_cfg},
      ),
      "actions": ObservationTermCfg(func=mdp.last_action),
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=1,
  )

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Pose rewards: only for legs + waist (arms are locked, no reward needed)
  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_LEGS_WAIST_JOINTS
  )
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
  }
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

  return cfg
