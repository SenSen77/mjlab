"""RL configuration for Unitree G1 23DOF tracking task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg


def unitree_g1_23_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  # Reuse the same PPO defaults as the 29DOF tracking task.
  return unitree_g1_tracking_ppo_runner_cfg()
