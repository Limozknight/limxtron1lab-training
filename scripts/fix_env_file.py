
import os

file_path = r"c:\Users\17950\Desktop\everything\IE\ACR\task234new\exts\bipedal_locomotion\bipedal_locomotion\tasks\locomotion\robots\limx_pointfoot_env_cfg.py"

# Define the new content to append
new_content = """

#############################
# 楼梯专项强化环境 / Stair Specialist Environment
# 适用于：1. 从零开始训练楼梯专家 (Foundation Training) 2. 基于已有模型微调 (Fine-tuning)
#############################

@configclass
class PFStairTrainingEnvCfg(PFTerrainTraversalEnvCfgV2):
    "楼梯专项训练环境 / Stair Specialist Environment"
    def __post_init__(self):
        super().__post_init__()
        
        # 1. 锁定地形为纯楼梯 / Lock terrain to stairs only
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG
        
        # 2. 难度设置 / Difficulty
        # Curriculum acts automatically: Starts at Diff 0 (5cm steps) -> Diff 1 (20cm steps)
        # 设定生成范围为全难度，课程管理器会自动从简单(难度0)开始让机器人爬
        self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        
        # 3. 奖励重点调整：爬楼梯需要更大的扭矩和更强的 Z 轴运动能力
        # [Tuning] Allow more torque for climbing
        self.rewards.pen_joint_torque.weight = -0.00005 
        # [Tuning] Allow vertical movement (lifting legs)
        self.rewards.pen_lin_vel_z.weight = -0.5 
        
        # 4. 降低速度要求：爬楼梯不求快，只求稳
        self.rewards.rew_lin_vel_xy_precise.weight = 3.0

@configclass
class PFStairTrainingEnvCfg_PLAY(PFStairTrainingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        # 测试时使用楼梯测试地形
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG
"""

# Read the file
with open(file_path, "rb") as f:
    content = f.read()

# Find the split point. 
# We look for the line: self.scene.terrain.terrain_generator = MIXED_TERRAINS_PLAY_CFG
marker = b"self.scene.terrain.terrain_generator = MIXED_TERRAINS_PLAY_CFG"
idx = content.find(marker)

if idx != -1:
    # Keep content up to the marker + length of marker
    clean_part = content[:idx + len(marker)]
    
    # Write back the clean part plus new content (encoded as utf-8)
    with open(file_path, "wb") as f:
        f.write(clean_part)
        f.write(new_content.encode("utf-8"))
    
    print("File fixed and updated successfully.")
else:
    print("Marker not found, manual check required.")
