import torch
import math
import collections
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import numpy as np

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

def linear_interpolation(start, end, t, max_t):
    return start + (end - start) * (t / max_t)

def get_from_curriculum(curriculum, t, max_t):
    min_start = curriculum["start"][0]
    min_end = curriculum["end"][0]
    max_start = curriculum["start"][1]
    max_end = curriculum["end"][1]
    min_value = linear_interpolation(min_start, min_end, t, max_t)
    max_value = linear_interpolation(max_start, max_end, t, max_t)
    return np.random.uniform(min_value, max_value)

class ZerothEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="mps"):
        self.device = torch.device(device)
        self.total_steps = 0
        self.max_steps = 40_000_000
        self.num_envs = num_envs
        self.num_single_obs = obs_cfg["num_single_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        
        # observation history
        self.frame_stack = obs_cfg.get("frame_stack", 1)
        self.num_obs = self.num_single_obs * self.frame_stack
        self.c_frame_stack = obs_cfg.get("c_frame_stack", 1)
        self.obs_history = collections.deque(maxlen=self.frame_stack)
        self.critic_history = collections.deque(maxlen=self.c_frame_stack)

        for _ in range(self.frame_stack):
            self.obs_history.append(
                torch.zeros(self.num_envs, self.num_single_obs, dtype=torch.float, device=self.device)
            )

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequence on real robot is 50Hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=4),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                gravity=(0, 0, -9.81)
            ),
            show_viewer=show_viewer,
        )
        
        self.horizontal_scale = 0.5
        terrain_length = 12.0
        half_terrain_length = terrain_length * 0.5
        # add terrain
        self.terrain = self.scene.add_entity(gs.morphs.Terrain(
            n_subterrains=(1, 1),
            subterrain_size=(terrain_length, terrain_length),
            pos=(-half_terrain_length, -half_terrain_length, 0.0), # align with the world origin
            randomize=True,
            subterrain_types="wave_terrain",
            horizontal_scale=self.horizontal_scale,
            vertical_scale=0.002,
            visualization=True,
            collision=True
        ))
        
        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="sim/resources/stompymicro/robot_fixed.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )
        
        # build
        self.scene.build(n_envs=num_envs, env_spacing=(1.0,1.0))

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]
        # Initialize legs_joints mapping with bounds checking
        self.legs_joints = {}
        joint_names = ["left_hip_pitch", "left_knee_pitch", "left_ankle_pitch",
                      "right_hip_pitch", "right_knee_pitch", "right_ankle_pitch"]
        
        for i, name in enumerate(joint_names):
            if i < len(self.motor_dofs):
                self.legs_joints[name] = self.motor_dofs[i] - 6 # 0，1，2，3，4，5， 6，7，8，9
            else:
                print(f"Warning: Joint {name} not found in motor_dofs")
                
        # Get number of bodies in the robot
        self.num_bodies = self.robot.n_links

        self.rand_push_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.rand_push_torque = torch.zeros((self.num_envs, 3), device=self.device)
        self.env_frictions = torch.ones((self.num_envs,), device=self.device)

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -9.81], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        # observation buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)

        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        # Initialize ref_dof_pos
        self.ref_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1).to(self.device)
        self.extras = {
            "observations": {},
        }  # extra information for logging

        # Initialize missing variables
        self.default_joint_pd_target = self.default_dof_pos.clone()
        self.base_euler = torch.zeros((self.num_envs, 3), device=self.device)
        self.filtered_base_height = torch.zeros(self.num_envs, device=self.device)
        # Initialize terrain difficulty
        self.terrain_difficulty = torch.zeros(self.num_envs, device=self.device)
        self.difficulty_factors = {
            "random_uniform_terrain": 0.3,
        }
    
    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        # Update last actions before execution
        self.last_actions[:] = self.actions[:]
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # Update reference state before computing rewards
        self.compute_ref_state()
        
        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # Compute observations
        self.compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # obs, rewards, dones, infos
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf, self.extras

    def _get_phase(self):
        cycle_time = self.env_cfg.get("cycle_time", 1.0)
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1).to(self.device)

        scale_1 = self.env_cfg.get("target_joint_pos_scale", 0.1)
        scale_2 = 2 * scale_1
        
        # Validate joint indices before accessing ref_dof_pos
        def safe_update(joint_name, scale):
            if joint_name in self.legs_joints:
                idx = self.legs_joints[joint_name]
                if idx < self.ref_dof_pos.shape[1]:
                    self.ref_dof_pos[:, idx] += scale
                else:
                    print(f"Warning: Joint index {idx} for {joint_name} is out of bounds")
            else:
                print(f"Warning: Joint {joint_name} not found in legs_joints")

        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        safe_update("left_hip_pitch", sin_pos_l * scale_1)
        safe_update("left_knee_pitch", sin_pos_l * scale_2)
        safe_update("left_ankle_pitch", sin_pos_l * scale_1)

        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        safe_update("right_hip_pitch", sin_pos_r * scale_1)
        safe_update("right_knee_pitch", sin_pos_r * scale_2)
        safe_update("right_ankle_pitch", sin_pos_r * scale_1)

        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

    def compute_observations(self):
        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        self.command_input = torch.cat((sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"]
        dq = self.dof_vel * self.obs_scales["dof_vel"]

        obs_buf = torch.cat( # total 41 dim
            (
                self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
                q,  # 10D
                dq,  # 10D
                self.actions,  # 10D
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.base_euler * self.obs_scales["quat"],  # 3
            ),
            dim=-1,
        )

        self.obs_buf = obs_buf

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        # Sample uniform multipliers between min and max
        kp_mult = gs_rand_float(
            self.env_cfg["kp_multipliers"][0],
            self.env_cfg["kp_multipliers"][1],
            (1,),
            self.device
        )
        kd_mult = gs_rand_float(
            self.env_cfg["kd_multipliers"][0], 
            self.env_cfg["kd_multipliers"][1],
            (1,),
            self.device
        )

        # Apply multipliers to default values
        kp_values = torch.full(
            (self.num_actions,),
            self.env_cfg["kp"] * kp_mult.item(),
            device=self.device
        )
        kd_values = torch.full(
            (self.num_actions,),
            self.env_cfg["kd"] * kd_mult.item(), 
            device=self.device
        )

        # Set the PD gains
        self.robot.set_dofs_kp(kp_values, self.motor_dofs)
        self.robot.set_dofs_kv(kd_values, self.motor_dofs)
        
        # friction
        friction = get_from_curriculum(self.env_cfg["env_friction_range"], self.total_steps, self.max_steps)
        self.robot.set_friction(friction)
        
        # link mass
        link_mass_mult = get_from_curriculum(self.env_cfg["link_mass_multipliers"], self.total_steps, self.max_steps)
        for link in self.robot.links:
            link.set_mass(link.get_mass() * link_mass_mult)

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # 获取全局地形高度场
        terrain_hf = self.terrain.terrain_hf  # 地形高度场矩阵
        
        # 将基座坐标转换到地形局部坐标系（地形原点在(0,0)）
        terrain_local_pos = self.base_pos[:, :2]  # 直接使用全局坐标
        
        # 转换为高度场索引
        x_idx = (terrain_local_pos[:, 0] / self.horizontal_scale).long()
        y_idx = (terrain_local_pos[:, 1] / self.horizontal_scale).long()
        
        # 添加边界保护
        x_idx = torch.clamp(x_idx, 0, terrain_hf.shape[0]-1)
        y_idx = torch.clamp(y_idx, 0, terrain_hf.shape[1]-1)
        
        # 确保地形高度场是二维数组
        if not isinstance(terrain_hf, np.ndarray):
            terrain_hf = np.array(terrain_hf, dtype=np.float32)
        
        # 将地形高度场转换为PyTorch张量并移到GPU
        terrain_hf_tensor = torch.from_numpy(terrain_hf).float().to(self.device)
        
        # 确保索引在有效范围内
        x_indices = torch.clamp(x_idx, 0, terrain_hf_tensor.shape[0]-1).long()
        y_indices = torch.clamp(y_idx, 0, terrain_hf_tensor.shape[1]-1).long()
        
        # 使用张量索引保持设备一致性
        terrain_height = terrain_hf_tensor[x_indices, y_indices]
        
        # 确保结果维度正确
        if terrain_height.dim() == 0:
            terrain_height = terrain_height.unsqueeze(0)
        
        # 计算基座相对高度
        base_height_above_terrain = self.base_pos[:, 2] - terrain_height
        height_error = torch.abs(base_height_above_terrain - self.reward_cfg["base_height_target"])
        
        return torch.exp(-height_error * self.reward_scales["base_height"])
    
    def _reward_gait_symmetry(self):
        # Reward symmetric gait patterns
        left_hip = self.dof_pos[:, self.env_cfg["dof_names"].index("left_hip_pitch")]
        right_hip = self.dof_pos[:, self.env_cfg["dof_names"].index("right_hip_pitch")]
        left_knee = self.dof_pos[:, self.env_cfg["dof_names"].index("left_knee_pitch")]
        right_knee = self.dof_pos[:, self.env_cfg["dof_names"].index("right_knee_pitch")]
        
        hip_symmetry = torch.abs(left_hip - right_hip)
        knee_symmetry = torch.abs(left_knee - right_knee)
        
        return torch.exp(-(hip_symmetry + knee_symmetry))
