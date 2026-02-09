"""Unitree G1 23DOF velocity environment configurations.

This task uses the rl_sar-aligned 23DOF MJCF asset and keeps the policy interface
consistent with `G1_23_JOINT_NAMES`.

Important differences from mjlab's 29DOF G1 velocity task:
- The 23DOF MJCF does not provide the same foot sites/geom naming, so foot-related
  observation/reward terms are implemented using body links (ankle roll links).
- Base velocity observations are derived from simulator state instead of relying on
  built-in sensor naming.
"""

from mjlab.asset_zoo.robots.unitree_g1_23.g1_23_constants import (
    G1_23_ACTION_SCALE,
    G1_23_JOINT_NAMES,
    get_g1_23_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.observations import body_height
from mjlab.tasks.velocity.mdp.rewards import (
    feet_clearance_bodies,
    feet_slip_bodies,
    feet_swing_height_bodies,
)
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

_FEET_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_JOINT_ASSET_CFG = SceneEntityCfg(
    "robot", joint_names=G1_23_JOINT_NAMES, preserve_order=True
)


def unitree_g1_23_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 23DOF rough terrain velocity configuration."""
    cfg = make_velocity_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 45

    cfg.scene.entities = {"robot": get_g1_23_robot_cfg()}

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

    if (
        cfg.scene.terrain is not None
        and cfg.scene.terrain.terrain_generator is not None
    ):
        cfg.scene.terrain.terrain_generator.curriculum = True

    # Action scaling: align with rl_sar (0.25 * effort_limit / stiffness per joint).
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_23_ACTION_SCALE

    cfg.viewer.body_name = "torso_link"

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 1.15

    # Observations: avoid depending on builtin sensor names (which differ across MJCFs).
    cfg.observations["policy"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel
    )
    cfg.observations["policy"].terms["base_ang_vel"] = ObservationTermCfg(
        func=mdp.base_ang_vel
    )
    cfg.observations["policy"].terms["joint_pos"].params["asset_cfg"] = _JOINT_ASSET_CFG
    cfg.observations["policy"].terms["joint_vel"].params["asset_cfg"] = _JOINT_ASSET_CFG

    cfg.observations["critic"].terms["base_lin_vel"] = cfg.observations["policy"].terms[
        "base_lin_vel"
    ]
    cfg.observations["critic"].terms["base_ang_vel"] = cfg.observations["policy"].terms[
        "base_ang_vel"
    ]
    cfg.observations["critic"].terms["joint_pos"].params["asset_cfg"] = _JOINT_ASSET_CFG
    cfg.observations["critic"].terms["joint_vel"].params["asset_cfg"] = _JOINT_ASSET_CFG

    # Replace foot-height critic term (sites) with body-link height.
    cfg.observations["critic"].terms["foot_height"] = ObservationTermCfg(
        func=body_height,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=_FEET_BODY_NAMES)},
    )

    # Randomization: the base velocity config relies on foot geom names that don't
    # exist in the rl_sar 23DOF MJCF. Disable this randomization for this task.
    cfg.events.pop("foot_friction", None)
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

    # Pose reward: MUST provide std mappings for all matched joints, otherwise
    # `variable_posture` will build an empty std tensor and crash.
    cfg.rewards["pose"].params["asset_cfg"] = _JOINT_ASSET_CFG
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
        # Lower body.
        r".*hip_pitch.*": 0.3,
        r".*hip_roll.*": 0.15,
        r".*hip_yaw.*": 0.15,
        r".*knee.*": 0.35,
        r".*ankle_pitch.*": 0.25,
        r".*ankle_roll.*": 0.1,
        # Waist.
        r".*waist_yaw.*": 0.2,
        # Arms.
        r".*shoulder_pitch.*": 0.15,
        r".*shoulder_roll.*": 0.15,
        r".*shoulder_yaw.*": 0.1,
        r".*elbow.*": 0.15,
        r".*wrist.*": 0.3,
    }
    cfg.rewards["pose"].params["std_running"] = {
        # Lower body.
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.2,
        r".*hip_yaw.*": 0.2,
        r".*knee.*": 0.6,
        r".*ankle_pitch.*": 0.35,
        r".*ankle_roll.*": 0.15,
        # Waist.
        r".*waist_yaw.*": 0.3,
        # Arms.
        r".*shoulder_pitch.*": 0.5,
        r".*shoulder_roll.*": 0.2,
        r".*shoulder_yaw.*": 0.15,
        r".*elbow.*": 0.35,
        r".*wrist.*": 0.3,
    }

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

    # Foot rewards: swap site-based terms for body-based terms.
    feet_asset_cfg = SceneEntityCfg("robot", body_names=_FEET_BODY_NAMES)
    cfg.rewards["foot_clearance"].func = feet_clearance_bodies
    cfg.rewards["foot_clearance"].params["asset_cfg"] = feet_asset_cfg

    cfg.rewards["foot_swing_height"].func = feet_swing_height_bodies
    cfg.rewards["foot_swing_height"].params["asset_cfg"] = feet_asset_cfg

    cfg.rewards["foot_slip"].func = feet_slip_bodies
    cfg.rewards["foot_slip"].params["asset_cfg"] = feet_asset_cfg

    # No air-time reward by default for this task.
    cfg.rewards["air_time"].weight = 0.0

    # Angular-momentum term depends on a builtin sensor name; keep it disabled.
    cfg.rewards["angular_momentum"].weight = 0.0

    # Add self-collision penalty (keep parity with base G1 velocity task style).
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Apply play mode overrides.
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )

        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def unitree_g1_23_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 23DOF flat terrain velocity configuration."""
    cfg = unitree_g1_23_rough_env_cfg(play=play)

    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    # Switch to flat terrain.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Disable terrain curriculum.
    assert "terrain_levels" in cfg.curriculum
    del cfg.curriculum["terrain_levels"]

    if play:
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
        twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

    return cfg
