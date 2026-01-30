"""Interface checks for the G1 23-DOF tracking task."""

import tempfile

import numpy as np
import pytest
from conftest import get_test_device

from mjlab.asset_zoo.robots.unitree_g1_23.g1_23_constants import get_g1_23_robot_cfg
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def test_g1_23dof_tracking_obs_action_shapes(device) -> None:
  # Build a dummy motion file with shapes consistent with the 23DOF robot.
  robot = Entity(get_g1_23_robot_cfg())
  num_joints = len(robot.joint_names)
  num_bodies = len(robot.body_names)

  T = 2  # minimal time steps
  joint_pos = np.zeros((T, num_joints), dtype=np.float32)
  joint_vel = np.zeros((T, num_joints), dtype=np.float32)

  body_pos_w = np.zeros((T, num_bodies, 3), dtype=np.float32)
  body_quat_w = np.zeros((T, num_bodies, 4), dtype=np.float32)
  body_quat_w[..., 0] = 1.0  # identity quats
  body_lin_vel_w = np.zeros((T, num_bodies, 3), dtype=np.float32)
  body_ang_vel_w = np.zeros((T, num_bodies, 3), dtype=np.float32)

  with tempfile.TemporaryDirectory() as tmpdir:
    motion_path = f"{tmpdir}/motion.npz"
    np.savez(
      motion_path,
      fps=np.array([200.0], dtype=np.float32),
      joint_pos=joint_pos,
      joint_vel=joint_vel,
      body_pos_w=body_pos_w,
      body_quat_w=body_quat_w,
      body_lin_vel_w=body_lin_vel_w,
      body_ang_vel_w=body_ang_vel_w,
    )

    cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1-23DoF")
    cfg.scene.num_envs = 1
    cfg.commands["motion"].motion_file = motion_path  # type: ignore[assignment]

    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    try:
      obs, _ = env.reset()
      assert "policy" in obs
      assert obs["policy"].shape == (1, 124)

      # Action space is 23-DOF.
      assert env.action_manager.action.shape == (1, 23)

      motion_term = env.command_manager.get_term("motion")
      assert isinstance(motion_term, MotionCommand)
      assert motion_term.command.shape == (1, 46)
    finally:
      env.close()
