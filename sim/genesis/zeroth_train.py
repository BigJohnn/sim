import argparse
import os
import pickle
import shutil

from zeroth_env import ZerothEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "num_steps_per_env": 48,
        "save_interval": 10,
        "empirical_normalization": True, #这个功能可以帮助稳定训练过程，特别是在观测值范围变化较大的情况下
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
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
        }
    }

    return train_cfg_dict


def get_cfgs():
    default_joint_angles={  # [rad]
            "left_elbow_yaw": 3.14,
            "right_elbow_yaw": 3.14,
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
        "num_actions": 12,  # 动作的数量
        # joint/link names
        "default_joint_angles": default_joint_angles,  # 默认关节角度
        "dof_names": list(default_joint_angles.keys()),  # 关节名称列表
        # PD
        "kp": 20.0,  # 比例增益（Proportional gain）
        "kd": 0.5,  # 微分增益（Derivative gain）
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
        "clip_actions": 100.0,  # 动作裁剪阈值
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "base_height_target": 0.32,  # Robot height
        "min_dist": 0.03,
        "max_dist": 0.14,
        "target_joint_pos_scale": 0.17,  # rad
        "target_feet_height": 0.02,  # m
        "cycle_time": 0.4,  # sec
        "only_positive_rewards": True,
        "tracking_sigma": 0.25,
        "max_contact_force": 100,  # forces above this value are penalized
        "reward_scales": {
            # reference motion tracking
            "joint_pos": 1.6,
            "feet_clearance": 1.5,
            "feet_contact_number": 1.5,
            "feet_air_time": 1.4,
            "foot_slip": -0.1,
            "feet_distance": 0.2,
            "knee_distance": 0.2,
            # contact
            "feet_contact_forces": -0.01,
            # vel tracking
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "vel_mismatch_exp": 0.5,  # lin_z; ang x,y
            "low_speed": 0.4,
            "track_vel_hard": 0.5,
            # base pos
            "default_joint_pos": 1.0,
            "orientation": 1,
            "base_height": 0.2,
            "base_acc": 0.2,
            # energy
            "action_smoothness": -0.002,
            "torques": -1e-5,
            "dof_vel": -5e-4,
            "dof_acc": -1e-7,
            "collision": -1.0,
        },
    }
    command_cfg = {
        "num_commands": 4,
        # "lin_vel_y_range": [-0.5, -0.5], # move forward slowly
        "lin_vel_y_range": [-0.6, -0.6], # move faster than above!
        "lin_vel_x_range": [-0.01, 0.01],
        "ang_vel_range": [-0.01, 0.01],
        "heading": [-3.14, 3.14]
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="zeroth-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
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
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    # runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="mps")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/zeroth_train.py
"""
