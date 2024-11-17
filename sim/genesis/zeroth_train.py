import argparse
import os
import pickle
import shutil

from zeroth_env import ZerothEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "num_steps_per_env": 64,  # 增加步数以收集更多样本
        "save_interval": 10,
        "empirical_normalization": True,
        "algorithm": {
            "rnd_cfg": None,
            "symmetry_cfg": None,
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.02,  # 增加熵系数以鼓励探索
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0005,  # 降低学习率以提高稳定性
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 8,  # 增加mini-batch数量
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
            "class_name": "PPO",
        },
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "experiment_name": exp_name,
            "run_name": "zeroth-walking",
            "device": "mps",
            "checkpoint_interval": 5,  # 每5次迭代保存checkpoint
            "max_checkpoints": 10  # 最多保留3个最新checkpoint
        }
    }

    return train_cfg_dict


def get_cfgs():
    default_joint_angles={  # [rad]
            "right_hip_pitch": 0.0,
            "left_hip_pitch": 0.0,
            "right_hip_yaw": 0.0,
            "left_hip_yaw": 0.0,
            "right_hip_roll": 0.0,
            "left_hip_roll": 0.0,
            "right_knee_pitch": 0.0,
            "left_knee_pitch": 0.0,
            "right_ankle_pitch": 0.0,
            "left_ankle_pitch": 0.0,
        }
    env_cfg = {
        "num_actions": 10,  # 动作的数量
        # joint/link names
        "default_joint_angles": default_joint_angles,  # 默认关节角度
        "dof_names": list(default_joint_angles.keys()),  # 关节名称列表
        # friction
        "env_friction_range": {
            "start": [0.9, 1.1],
            "end": [0.9, 1.1],
        },
        # link mass
        # varying this too much collapses the training
        "link_mass_multipliers": {
            "start": [1.0, 1.0],
            "end": [1.0, 1.0],
        },
        # RFI
        "rfi_scale": 0.1,
        # PD
        "kp": 20.0,  # 比例增益（Proportional gain）
        "kd": 0.5,  # 微分增益（Derivative gain）
        "kp_multipliers": [0.75, 1.25],
        "kd_multipliers": [0.75, 1.25],
        # termination
        "termination_if_roll_greater_than": 10,  # 如果滚转角度大于10度则终止
        "termination_if_pitch_greater_than": 10,  # 如果俯仰角度大于10度则终止
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],  # 基础初始位置
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],  # 基础初始四元数（表示旋转）
        "episode_length_s": 20.0,  # 每个训练回合的时长（秒）
        "resampling_time_s": 4.0,  # 重新采样时间间隔（秒）
        "action_scale": 0.25,  # 动作缩放比例
        "simulate_action_latency": True,  # 是否模拟动作延迟
        "clip_actions": 18.0,  # 动作裁剪阈值
    }
    obs_cfg = {
        "num_single_obs": 41,
        "obs_scales": {
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "ang_vel": 1.0,
            "lin_vel": 2.0,
            "quat": 1.0,
        },
    }
    reward_cfg = {
        "base_height_target": 0.32,
        "min_dist": 0.03,
        "max_dist": 0.14,
        "target_joint_pos_scale": 0.17,
        "target_feet_height": 0.02,
        "cycle_time": 0.4,
        "only_positive_rewards": True,
        "tracking_sigma": 0.25,
        "max_contact_force": 100,
        "reward_scales": {
            "tracking_lin_vel": 1.2,
            "tracking_ang_vel": 0.3,
            "lin_vel_z": -1.5,
            "action_rate": -0.01,
            "similar_to_default": -0.2,
            "base_height": 0.3,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_y_range": [-0.6, -0.6], # move faster than above!
        "lin_vel_x_range": [-0.01, 0.01],
        "ang_vel_range": [-0.01, 0.01],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg

def run_learner(runner, max_iterations):
    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="zeroth-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("-V", "--vis", type=bool, default=False)
    parser.add_argument("--max_iterations", type=int, default=101)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = ZerothEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,show_viewer=args.vis,
    )

    # runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="mps")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    gs.tools.run_in_another_thread(fn=run_learner, args=(runner, args.max_iterations)) # start the simulation in another thread
    if args.vis:
        env.scene.viewer.start() # start the viewer in the main thread (the render thread)

if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/zeroth_train.py
"""
