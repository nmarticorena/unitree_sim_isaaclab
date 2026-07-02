# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
from isaaclab.assets.asset_base_cfg import AssetBaseCfg
import os
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
import isaaclab.sim as sim_utils
from . import mdp
# use Isaac Lab native event system

from tasks.common_config import  G1RobotPresets, CameraPresets  # isort: skip
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager

# import public scene configuration
from tasks.common_scene.base_scene_empty_cfg import EmptySceneCfg

##
# Scene definition
##

project_root = os.environ.get("PROJECT_ROOT")
DOLLY_ROOT_Z = 0.52
DOLLY_DEFAULT_POS = (-4.75, -3.25, DOLLY_ROOT_Z)
DOLLY_RANDOM_POSE_RANGE = {"x": (-3.5, -3), "y": (-3.5, -3.0), "z": (DOLLY_ROOT_Z, DOLLY_ROOT_Z)}
DOLLY_RANDOM_VELOCITY_RANGE = {}


def _resolve_env_ids(env, env_ids):
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    return env_ids


def _sample_uniform_ranges(range_cfg, keys, default_values, device):
    samples = default_values.clone()
    for index, key in enumerate(keys):
        if key in range_cfg:
            low, high = range_cfg[key]
            samples[:, index] = torch.empty(samples.shape[0], device=device).uniform_(float(low), float(high))
    return samples


def _reset_dolly_random(env, env_ids):
    env_ids = _resolve_env_ids(env, env_ids)
    asset = env.scene["dolly"]
    root_states = asset.data.default_root_state[env_ids].clone()

    positions = _sample_uniform_ranges(
        DOLLY_RANDOM_POSE_RANGE,
        ("x", "y", "z"),
        root_states[:, 0:3],
        asset.device,
    )
    positions += env.scene.env_origins[env_ids]
    velocities = _sample_uniform_ranges(
        DOLLY_RANDOM_VELOCITY_RANGE,
        ("x", "y", "z", "roll", "pitch", "yaw"),
        root_states[:, 7:13],
        asset.device,
    )
    asset.write_root_pose_to_sim(torch.cat([positions, root_states[:, 3:7]], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def _reset_scene_with_random_dolly(env):
    env_ids = torch.arange(env.num_envs, device=env.device)
    base_mdp.reset_scene_to_default(env, env_ids)
    return _reset_dolly_random(env, env_ids)


@configclass
class ObjectTableSceneCfg(EmptySceneCfg):
    """object table scene configuration class
    inherits from G1SingleObjectSceneCfg, gets the complete G1 robot scene configuration
    can add task-specific scene elements or override default configurations here
    """
    
    # Humanoid robot w/ arms higher
    # 5. humanoid robot configuration 
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_inspire_wholebody(init_pos=(-3.9, -2.81811, 0.8),
        init_rot=(1, 0, 0, 0))

    dolly: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Dolly",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=DOLLY_DEFAULT_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{project_root}/assets/dolly/dolly_rigid.usd",

            # Important: disable articulation roots created by the MJCF importer.
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),

            # Make sure it behaves as a dynamic rigid object.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),

            mass_props=sim_utils.MassPropertiesCfg(
                mass=3.0,
            ),

            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02,
                rest_offset=0.0,
            ),
        ),
)

    contact_forces = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", history_length=10, track_air_time=True, debug_vis=False)
    # 6. add camera configuration 
    front_camera = CameraPresets.g1_front_camera()
    # left_wrist_camera = CameraPresets.left_inspire_wrist_camera()
    # right_wrist_camera = CameraPresets.right_inspire_wrist_camera()
    robot_camera = CameraPresets.g1_world_camera()
##
# MDP settings
##
@configclass
class ActionsCfg:
    """defines the action configuration related to robot control, using direct joint angle control
    """
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)



@configclass
class ObservationsCfg:
    """
    defines all available observation information
    """
    @configclass
    class PolicyCfg(ObsGroup):
        """policy group observation configuration class
        defines all state observation values for policy decision
        inherit from ObsGroup base class 
        """

        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_inspire_state = ObsTerm(func=mdp.get_robot_inspire_joint_states)
        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            """post initialization function
            set the basic attributes of the observation group
            """
            self.enable_corruption = False  # disable observation value corruption
            self.concatenate_terms = False  # disable observation item connection

    # observation groups
    # create policy observation group instance
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    pass
    # check if the object is out of the working range
    # success = DoneTerm(func=mdp.reset_object_estimate)# use task completion check function

@configclass
class RewardsCfg:
    reward = RewTerm(func=mdp.compute_reward, weight=1.0, params={"object_cfg": SceneEntityCfg("dolly")})

@configclass
class EventCfg:
    reset_dolly = EventTermCfg(
        func=_reset_dolly_random,
        mode="reset",
    )


@configclass
class PushCartG129InspireWholebodyEnvCfg(ManagerBasedRLEnvCfg):
    """
    inherits from ManagerBasedRLEnvCfg, defines all configuration parameters for the entire environment
    """

    # 1. scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, # environment number: 1
                                                     env_spacing=2.5, # environment spacing: 2.5 meter
                                                     replicate_physics=True # enable physics replication
                                                     )
    # basic settings
    observations: ObservationsCfg = ObservationsCfg()   # observation configuration
    actions: ActionsCfg = ActionsCfg()                  # action configuration
    # MDP settings
        
    terminations: TerminationsCfg = TerminationsCfg()    # termination configuration
    events = EventCfg()                                  # event configuration
    commands = None # command manager
    rewards: RewardsCfg = RewardsCfg()  # reward manager
    curriculum = None # curriculum manager
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

                # Physics material properties
        self.sim.physics_material.static_friction = 1.0  # Static friction
        self.sim.physics_material.dynamic_friction = 1.0  # Dynamic friction
        self.sim.physics_material.friction_combine_mode = "max"  # Friction combine mode
        self.sim.physics_material.restitution_combine_mode = "max"  # Restitution combine mode
        # create event manager
        self.event_manager = SimpleEventManager()

        # register "reset dolly" event
        self.event_manager.register("reset_object_self", SimpleEvent(
            func=lambda env: _reset_dolly_random(env, torch.arange(env.num_envs, device=env.device))
        ))
        
        self.event_manager.register("reset_all_self", SimpleEvent(
            func=_reset_scene_with_random_dolly
        ))
