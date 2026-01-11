
import os

file_path = r"c:\Users\17950\Desktop\everything\IE\ACR\task234new\exts\bipedal_locomotion\bipedal_locomotion\tasks\locomotion\robots\limx_pointfoot_env_cfg.py"

# New content to append (Stairs Environment)
new_content = """

#############################
# 楼梯专项强化环境 / Stair Specialist Environment
#############################

@configclass
class PFStairTrainingEnvCfg(PFTerrainTraversalEnvCfgV2):
    "楼梯专项训练环境 / Stair Specialist Environment"
    def __post_init__(self):
        super().__post_init__()
        
        # 1. 锁定地形为纯楼梯 / Lock terrain to stairs only
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG
        
        # 2. 难度设置 / Difficulty
        # 设定生成范围为全难度，课程管理器会自动从简单(难度0)开始
        # Start at 0.0 (Easy) -> 1.0 (Hard)
        self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        
        # 3. 奖励重点调整 / Reward Tuning
        # Allow more torque for climbing
        self.rewards.pen_joint_torque.weight = -0.00005 
        # Allow vertical movement (lifting legs)
        self.rewards.pen_lin_vel_z.weight = -0.5 
        
        # 4. 降低速度要求 / Lower speed requirements
        self.rewards.rew_lin_vel_xy_precise.weight = 3.0

@configclass
class PFStairTrainingEnvCfg_PLAY(PFStairTrainingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        # 测试时使用楼梯测试地形 / Test on stairs
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG
"""

try:
    with open(file_path, "rb") as f:
        content = f.read()
    
    # Locate the end of the previous known class 'PFUnifiedEnvCfg_PLAY'
    # We look for the assignment to MIXED_TERRAINS_PLAY_CFG
    marker = b"self.scene.terrain.terrain_generator = MIXED_TERRAINS_PLAY_CFG"
    idx = content.find(marker)
    
    if idx == -1:
        print("Error: Could not find the marker 'MIXED_TERRAINS_PLAY_CFG'. File structure might be unexpected.")
    else:
        # Calculate cut-off point: marker index + marker length
        cutoff = idx + len(marker)
        
        # Keep everything before the cutoff
        clean_content = content[:cutoff]
        
        # Write back the clean content + new content
        with open(file_path, "wb") as f:
            f.write(clean_content)
            f.write(new_content.encode('utf-8'))
            
        print("Success: File truncated and new content appended.")

except Exception as e:
    print(f"An error occurred: {e}")
