"""Unitree G1 23-DOF constants (rl_sar compatible).

This robot definition intentionally reuses the 23-DOF MJCF from rl_sar and keeps the
asset files tracked inside mjlab so it can be synced to training servers.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  FEET_ONLY_COLLISION,
  FULL_COLLISION_WITHOUT_SELF,
  G1_ACTUATOR_5020,
  G1_ACTUATOR_7520_14,
  G1_ACTUATOR_7520_22,
  G1_ACTUATOR_ANKLE,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF and assets.
##

G1_23_MJCF: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_g1_23" / "mjcf" / "g1_23dof.xml"
)
assert G1_23_MJCF.exists()


def get_assets(meshdir: str | None) -> dict[str, bytes]:
  """Load mesh assets into the MjSpec assets dict."""
  assets: dict[str, bytes] = {}
  mesh_root = G1_23_MJCF.parent / (meshdir or "")
  if mesh_root.exists():
    update_assets(assets, mesh_root, meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_23_MJCF))
  # Bundle meshes so compilation doesn't depend on CWD/FS layout.
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Joint interface (must match rl_sar/policy/g1_23/base.yaml exactly).
##

G1_23_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
)


_G1_23_DEFAULT_DOF_POS = (
  -0.20,  # left_hip_pitch_joint
  0.0,  # left_hip_roll_joint
  0.0,  # left_hip_yaw_joint
  0.42,  # left_knee_joint
  -0.23,  # left_ankle_pitch_joint
  0.0,  # left_ankle_roll_joint
  -0.20,  # right_hip_pitch_joint
  0.0,  # right_hip_roll_joint
  0.0,  # right_hip_yaw_joint
  0.42,  # right_knee_joint
  -0.23,  # right_ankle_pitch_joint
  0.0,  # right_ankle_roll_joint
  0.0,  # waist_yaw_joint
  0.35,  # left_shoulder_pitch_joint
  0.16,  # left_shoulder_roll_joint
  0.0,  # left_shoulder_yaw_joint
  0.87,  # left_elbow_joint
  0.0,  # left_wrist_roll_joint
  0.35,  # right_shoulder_pitch_joint
  -0.16,  # right_shoulder_roll_joint
  0.0,  # right_shoulder_yaw_joint
  0.87,  # right_elbow_joint
  0.0,  # right_wrist_roll_joint
)

_G1_23_DEFAULT_JOINT_POS = dict(zip(G1_23_JOINT_NAMES, _G1_23_DEFAULT_DOF_POS))


##
# Actuator config.
##

# We reuse the same actuator parameterization as mjlab's G1, but only include
# the groups that exist for 23-DOF (no waist pitch/roll, no wrist pitch/yaw).
G1_23_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


G1_23_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.76),
  joint_pos={".*": 0.0, **_G1_23_DEFAULT_JOINT_POS},
  joint_vel={".*": 0.0},
)


def get_g1_23_robot_cfg() -> EntityCfg:
  """Get a fresh G1-23DOF robot configuration instance."""
  return EntityCfg(
    init_state=G1_23_INIT_STATE,
    # Collisions are still configured via spec editors if matching geoms exist.
    collisions=(FULL_COLLISION_WITHOUT_SELF, FEET_ONLY_COLLISION),
    spec_fn=get_spec,
    articulation=G1_23_ARTICULATION,
  )


G1_23_ACTION_SCALE: dict[str, float] = {}
for a in G1_23_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    G1_23_ACTION_SCALE[n] = 0.25 * e / s
