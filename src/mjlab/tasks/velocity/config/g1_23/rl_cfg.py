"""RL configuration for Unitree G1 23DOF velocity task."""

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg


def unitree_g1_23_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 23DOF velocity task."""
  base = unitree_g1_ppo_runner_cfg()
  return replace(base, experiment_name="g1_23_velocity")
