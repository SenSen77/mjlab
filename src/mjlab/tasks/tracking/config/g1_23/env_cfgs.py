"""Unitree G1 23-DOF flat tracking environment configurations."""

from mjlab.asset_zoo.robots.unitree_g1_23.g1_23_constants import (
  G1_23_ACTION_SCALE,
  G1_23_JOINT_NAMES,
  get_g1_23_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def unitree_g1_23_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 23DOF flat terrain tracking configuration.

  This configuration is aligned to rl_sar's `policy/g1_23` interface:
  - obs dim: 124 = 46 (motion_command) + 6 (motion_anchor_ori_b) + 3 (ang_vel)
            + 23 (dof_pos) + 23 (dof_vel) + 23 (actions)
  - action dim: 23
  """
  cfg = make_tracking_env_cfg()

  # Use the 23DOF robot MJCF.
  cfg.scene.entities = {"robot": get_g1_23_robot_cfg()}

  # Self-collision sensor (optional, but keep parity with the base tracking task).
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  # Action scaling: align with rl_sar (0.25 * effort_limit / stiffness per joint).
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_23_ACTION_SCALE

  # Motion command: only include the 23DOF joint subset in the command vector (46-dim).
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.command_joint_names = G1_23_JOINT_NAMES
  motion_cmd.anchor_body_name = "torso_link"
  # Body set used for body-position/orientation rewards & debugging.
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_rubber_hand",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_rubber_hand",
  )

  # The base tracking config contains a foot friction randomization that relies on
  # specific named foot collision geoms in mjlab's 29DOF MJCF. The rl_sar 23DOF MJCF
  # doesn't provide those names, so disable this randomization for this task.
  cfg.events.pop("foot_friction", None)

  # Base COM randomization should reference a valid body.
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # EE termination bodies: use ankles and rubber-hand end effectors.
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
  )

  cfg.viewer.body_name = "torso_link"

  # Replace the default policy observation layout to match rl_sar exactly (124-dim).
  # We intentionally do not use state-estimation dependent terms for this interface.
  # NOTE: `mdp.*` comes from `mjlab.tasks.tracking.mdp` which re-exports `mjlab.envs.mdp`.
  from mjlab.tasks.tracking import mdp

  joint_asset_cfg = SceneEntityCfg("robot", joint_names=G1_23_JOINT_NAMES)
  cfg.observations["policy"] = ObservationGroupCfg(
    terms={
      "command": ObservationTermCfg(
        func=mdp.generated_commands, params={"command_name": "motion"}
      ),
      "motion_anchor_ori_b": ObservationTermCfg(
        func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
      ),
      "ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
      "dof_pos": ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={"biased": True, "asset_cfg": joint_asset_cfg},
      ),
      "dof_vel": ObservationTermCfg(
        func=mdp.joint_vel_rel, params={"asset_cfg": joint_asset_cfg}
      ),
      "actions": ObservationTermCfg(func=mdp.last_action),
    },
    concatenate_terms=True,
    enable_corruption=has_state_estimation,
  )

  # Critic observations: avoid relying on XML sensor names that differ across MJCFs.
  # Use state-derived base velocities instead of builtin_sensor("robot/imu_*").
  cfg.observations["critic"] = ObservationGroupCfg(
    terms={
      "command": ObservationTermCfg(
        func=mdp.generated_commands, params={"command_name": "motion"}
      ),
      "motion_anchor_pos_b": ObservationTermCfg(
        func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
      ),
      "motion_anchor_ori_b": ObservationTermCfg(
        func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
      ),
      "body_pos": ObservationTermCfg(
        func=mdp.robot_body_pos_b, params={"command_name": "motion"}
      ),
      "body_ori": ObservationTermCfg(
        func=mdp.robot_body_ori_b, params={"command_name": "motion"}
      ),
      "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
      "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
      "joint_pos": ObservationTermCfg(
        func=mdp.joint_pos_rel, params={"asset_cfg": joint_asset_cfg}
      ),
      "joint_vel": ObservationTermCfg(
        func=mdp.joint_vel_rel, params={"asset_cfg": joint_asset_cfg}
      ),
      "actions": ObservationTermCfg(func=mdp.last_action),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
