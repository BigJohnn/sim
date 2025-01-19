import argparse
import os
import pickle

import torch
from zeroth_env import ZerothEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

def run_sim(env, policy, obs):
    while True:
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="zeroth-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    # Add missing class_name fields
    train_cfg["algorithm"]["class_name"] = "PPO"
    train_cfg["policy"]["class_name"] = "ActorCritic"
    print("train_cfg:", train_cfg)  # Add debug print
    reward_cfg["reward_scales"] = {}

    env = ZerothEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="mps")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    # policy = runner.get_inference_policy(device="cuda:0")
    policy = runner.get_inference_policy(device="mps")

    obs, _ = env.reset()
    with torch.no_grad():
        gs.tools.run_in_another_thread(fn=run_sim, args=(env, policy, obs)) # start the simulation in another thread
        env.scene.viewer.start() # start the viewer in the main thread (the render thread)

if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/zeroth_eval.py -e zeroth-walking -v --ckpt 100
"""
