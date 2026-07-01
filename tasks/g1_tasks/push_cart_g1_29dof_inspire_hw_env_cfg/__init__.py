
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  

import gymnasium as gym

from . import push_cart_g1_29dof_inspire_hw_env_cfg as config


gym.register(
    id="Isaac-Push-Cart-G129-Inspire-Wholebody",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": config.PushCartG129InspireWholebodyEnvCfg,
    },
    disable_env_checker=True,
)

