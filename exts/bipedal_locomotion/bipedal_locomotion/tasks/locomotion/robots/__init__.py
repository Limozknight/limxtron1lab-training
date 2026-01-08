import gymnasium as gym

from bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_ppo_cfg import PF_TRON1AFlatPPORunnerCfg, WF_TRON1AFlatPPORunnerCfg, SF_TRON1AFlatPPORunnerCfg

from . import limx_pointfoot_env_cfg, limx_wheelfoot_env_cfg, limx_solefoot_env_cfg

##
# Create PPO runners for RSL-RL
##

limx_pf_blind_flat_runner_cfg = PF_TRON1AFlatPPORunnerCfg()

limx_wf_blind_flat_runner_cfg = WF_TRON1AFlatPPORunnerCfg()

limx_sf_blind_flat_runner_cfg = SF_TRON1AFlatPPORunnerCfg()



##
# Register Gym environments
##

############################
# PF Blind Flat Environment
############################
gym.register(
    id="Isaac-Limx-PF-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-PF-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)

#############################
# PF Terrain Traversal (Task 2.4)
#############################
gym.register(
    id="Isaac-Limx-PF-Terrain-Traversal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFTerrainTraversalEnvCfg,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-PF-Terrain-Traversal-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFTerrainTraversalEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)

#############################
# PF Terrain Traversal V2 (Task 2.4 Optimized)
#############################
gym.register(
    id="Isaac-Limx-PF-Terrain-Traversal-V2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFTerrainTraversalEnvCfgV2,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-PF-Terrain-Traversal-V2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFTerrainTraversalEnvCfgV2_PLAY,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)

#############################
# PF Stair Traversal (Task 2.4 - Stairs)
#############################
gym.register(
    id="Isaac-Limx-PF-Stair-Traversal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFStairTraversalEnvCfg,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-PF-Stair-Traversal-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFStairTraversalEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)

#############################
# WF Blind Flat Environment
#############################
gym.register(
    id="Isaac-Limx-WF-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_env_cfg.WFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_env_cfg.WFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_blind_flat_runner_cfg,
    },
)


############################
# SF Blind Flat Environment
############################
gym.register(
    id="Isaac-Limx-SF-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

#############################
# Task 2.5: Pronk Environment
#############################
gym.register(
    id="Isaac-Limx-PF-Pronk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFPronkEnvCfg,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-PF-Pronk-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFPronkEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)