import torch
import math
import collections
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class ZerothEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="mps"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.add_noise = obs_cfg.get("add_noise", False)
        
        # observation history
        self.frame_stack = obs_cfg.get("frame_stack", 15)
        self.c_frame_stack = obs_cfg.get("c_frame_stack", 3)
        self.obs_history = collections.deque(maxlen=self.frame_stack)
        self.critic_history = collections.deque(maxlen=self.frame_stack)

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.001  # control frequence on real robot is 1000hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
        
        # privileged observation config
        self.num_privileged_obs = 66

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
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
            ),
            show_viewer=show_viewer,
        )

        # add terrain
        self.terrain = self.scene.add_entity(gs.morphs.Terrain(
            n_subterrains=(3, 3),
            subterrain_size=(12.0, 12.0),
            horizontal_scale=0.25,
            vertical_scale=0.005,
            subterrain_types=[
                ["flat_terrain", "random_uniform_terrain", "stepping_stones_terrain"],
                ["pyramid_sloped_terrain", "discrete_obstacles_terrain", "wave_terrain"], 
                ["random_uniform_terrain", "pyramid_stairs_terrain", "sloped_terrain"]
            ],
            visualization=True,
            collision=True
        ))
        
        # terrain measurement
        self.measure_heights = True
        self.measured_heights = None
        self.height_samples = 64

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
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # Initialize legs_joints mapping
        self.legs_joints = {
            "left_hip_pitch": self.motor_dofs[0],
            "left_knee_pitch": self.motor_dofs[1],
            "left_ankle_pitch": self.motor_dofs[2],
            "right_hip_pitch": self.motor_dofs[3],
            "right_knee_pitch": self.motor_dofs[4],
            "right_ankle_pitch": self.motor_dofs[5]
        }

        # Initialize feet indices
        self.feet_indices = [
            self.robot.get_link("foot_left").idx,
            self.robot.get_link("foot_right").idx,
        ]

        # Get number of bodies in the robot
        self.num_bodies = self.robot.n_links

        # Initialize observation related buffers
        self.contact_forces = torch.zeros((self.num_envs, len(self.feet_indices), 3), device=self.device)
        self.net_contact_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.rand_push_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.rand_push_torque = torch.zeros((self.num_envs, 3), device=self.device)
        self.env_frictions = torch.ones((self.num_envs,), device=self.device)
        self.body_mass = torch.ones((self.num_envs,), device=self.device) * 30.0

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
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        # observation buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs * self.frame_stack), device=self.device, dtype=gs.tc_float)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device, dtype=gs.tc_float)
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
        self.extras = dict()  # extra information for logging

        self.noise_scale_vec = self._get_noise_scale_vec(obs_cfg)

        # Initialize missing variables
        self.default_joint_pd_target = self.default_dof_pos.clone()
        self.base_euler = torch.zeros((self.num_envs, 3), device=self.device)
        # Initialize terrain difficulty
        self.terrain_difficulty = torch.zeros(self.num_envs, device=self.device)
        self.difficulty_factors = {
            "flat_terrain": 0.1,
            "random_uniform_terrain": 0.3,
            "stepping_stones_terrain": 0.5,
            "pyramid_sloped_terrain": 0.7,
            "discrete_obstacles_terrain": 0.8,
            "wave_terrain": 0.6,
            "pyramid_stairs_terrain": 0.9,
            "sloped_terrain": 0.4
        }
        self.root_states = torch.zeros((self.num_envs, 13), device=self.device)
        self.rigid_state = torch.zeros((self.num_envs, self.num_bodies, 13), device=self.device)
    
    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 3] = gs_rand_float(*self.command_cfg["heading"], (len(envs_idx),), self.device)

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

        # Get net contact forces for all bodies
        contact_forces = self.scene.get_contact_forces()
        for env_idx in range(self.num_envs):
            for body_idx in range(self.num_bodies):
                self.net_contact_forces[env_idx, body_idx] = torch.tensor(
                    contact_forces[env_idx][body_idx],
                    device=self.device
                )

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

        return self.obs_buf, self.rew_buf, self.reset_buf, {
            "observations": {
                "critic": self.privileged_obs_buf
            },
            **self.extras
        }

    def get_observations(self):
        return self.obs_buf, {"observations": {"critic": self.obs_buf}}

    def _get_phase(self):
        cycle_time = self.env_cfg.get("cycle_time", 1.0)
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1
        return stance_mask

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        default_clone = self.default_dof_pos.clone()
        self.ref_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)

        scale_1 = self.env_cfg.get("target_joint_pos_scale", 0.1)
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, self.legs_joints["left_hip_pitch"]] += sin_pos_l * scale_1
        self.ref_dof_pos[:, self.legs_joints["left_knee_pitch"]] += sin_pos_l * scale_2
        self.ref_dof_pos[:, self.legs_joints["left_ankle_pitch"]] += sin_pos_l * scale_1

        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, self.legs_joints["right_hip_pitch"]] += sin_pos_r * scale_1
        self.ref_dof_pos[:, self.legs_joints["right_knee_pitch"]] += sin_pos_r * scale_2
        self.ref_dof_pos[:, self.legs_joints["right_ankle_pitch"]] += sin_pos_r * scale_1

        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

        self.ref_action = 2 * self.ref_dof_pos

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        num_actions = self.num_actions
        noise_vec = torch.zeros(self.num_obs, device=self.device)
        self.add_noise = cfg.get("add_noise", False)
        noise_scales = cfg["noise_scales"]
        noise_vec[0:5] = 0.0  # commands
        noise_vec[5 : (num_actions + 5)] = noise_scales["dof_pos"] * self.obs_scales["dof_pos"]
        noise_vec[(num_actions + 5) : (2 * num_actions + 5)] = noise_scales["dof_vel"] * self.obs_scales["dof_vel"]
        noise_vec[(2 * num_actions + 5) : (3 * num_actions + 5)] = 0.0  # previous actions
        noise_vec[(3 * num_actions + 5) : (3 * num_actions + 5) + 3] = (
            noise_scales["ang_vel"] * self.obs_scales["ang_vel"]
        )  # ang vel
        noise_vec[(3 * num_actions + 5) + 3 : (3 * num_actions + 5)] = (
            noise_scales["quat"] * self.obs_scales["quat"]
        )  # euler x,y
        return noise_vec

    def compute_observations(self):
        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.0

        self.command_input = torch.cat((sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        diff = self.dof_pos - self.ref_dof_pos
        self.privileged_obs_buf = torch.cat( # total 66 dim
            (
                self.command_input,  # 2 + 3
                (self.dof_pos - self.default_joint_pd_target) * self.obs_scales.dof_pos,  # 10D
                self.dof_vel * self.obs_scales.dof_vel,  # 10D
                self.actions,  # 10D
                diff,  # 10D
                self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.base_euler * self.obs_scales.quat,  # 3
                self.rand_push_force[:, :2],  # 3
                self.rand_push_torque,  # 3
                self.env_frictions,  # 1
                self.body_mass / 30.0,  # 1
                stance_mask,  # 2
                contact_mask,  # 2
            ),
            dim=-1,
        )

        obs_buf = torch.cat( # total 43 dim
            (
                self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
                q,  # 10D
                dq,  # 10D
                self.actions,  # 10D
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.base_euler * self.obs_scales.quat,  # 3
                self.measured_heights.mean(dim=1, keepdim=True) * self.obs_scales.height_measurements,  # 1
                self.terrain_difficulty.unsqueeze(1),  # 1
            ),
            dim=-1,
        )

        if self.measure_heights:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            self.privileged_obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        if self.add_noise:
            noise_level = self.obs_cfg.get("noise_level", 0.1)
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * noise_level
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.c_frame_stack)], dim=1)

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset observation history
        for i in range(self.frame_stack):
            self.obs_history[i][envs_idx] *= 0
            if self.num_privileged_obs is not None:
                self.critic_history[i][envs_idx] *= 0

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
        stance_mask = self._get_gait_phase()
        self.measured_heights = torch.sum(self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(
            stance_mask, dim=1
        )
        # Penalize base height away from target
        height_error = torch.abs(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
        return torch.exp(-height_error * self.reward_cfg["reward_scales"]["base_height"])
    
    def _reward_gait_symmetry(self):
        # Reward symmetric gait patterns
        left_hip = self.dof_pos[:, self.env_cfg["dof_names"].index("left_hip_pitch")]
        right_hip = self.dof_pos[:, self.env_cfg["dof_names"].index("right_hip_pitch")]
        left_knee = self.dof_pos[:, self.env_cfg["dof_names"].index("left_knee_pitch")]
        right_knee = self.dof_pos[:, self.env_cfg["dof_names"].index("right_knee_pitch")]
        
        hip_symmetry = torch.abs(left_hip - right_hip)
        knee_symmetry = torch.abs(left_knee - right_knee)
        
        return torch.exp(-(hip_symmetry + knee_symmetry))

    def _reward_energy_efficiency(self):
        # Reward energy efficiency by penalizing high joint velocities
        return self.reward_cfg["reward_scales"]["dof_vel"] * torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_orientation(self):
        # Penalize base orientation away from upright
        return torch.exp(-torch.abs(self.base_euler[:, 0]) - torch.abs(self.base_euler[:, 1])) * self.reward_cfg["reward_scales"]["orientation"]

    def _reward_terrain_adaptation(self):
        """Reward for adapting to different terrain types"""
        # Calculate foot clearance
        foot_clearance = torch.zeros((self.num_envs, 2), device=self.device)
        for i, foot_idx in enumerate(self.feet_indices):
            foot_pos = self.robot.get_link(id=foot_idx).get_pos()
            terrain_height = self.terrain.get_height(foot_pos[:, :2])
            foot_clearance[:, i] = foot_pos[:, 2] - terrain_height

        # Reward for maintaining appropriate foot clearance
        target_clearance = 0.05  # 5cm
        clearance_error = torch.abs(foot_clearance - target_clearance)
        return torch.exp(-torch.mean(clearance_error, dim=1) / 0.02)

    def _reward_terrain_stability(self):
        """Reward for maintaining stability on uneven terrain"""
        # Penalize large base orientation changes
        base_euler = quat_to_xyz(self.base_quat)
        return torch.exp(-torch.abs(base_euler[:, 0]) - torch.abs(base_euler[:, 1]))

    def _reward_terrain_progress(self):
        """Reward for making progress across different terrain types"""
        # Calculate forward progress relative to terrain difficulty
        forward_vel = self.base_lin_vel[:, 0]
        
        # Difficulty factors based on terrain type
        difficulty_factors = {
            "flat_terrain": 0.1,
            "random_uniform_terrain": 0.3,
            "stepping_stones_terrain": 0.5,
            "pyramid_sloped_terrain": 0.7,
            "discrete_obstacles_terrain": 0.8,
            "wave_terrain": 0.6,
            "pyramid_stairs_terrain": 0.9,
            "sloped_terrain": 0.4
        }
        
        # Get current terrain type
        terrain_type = self.terrain.get_type(self.base_pos[:, :2])
        difficulty = difficulty_factors.get(terrain_type, 0.5)
        
        return forward_vel / (difficulty + 0.1)

    def _reward_joint_pos(self):
        """Calculates the reward based on the difference between the current joint positions and the target joint positions."""
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)

        return r

    def _reward_contact_stability(self):
        """Reward for maintaining stable contact force distribution"""
        # Get contact forces for feet
        left_foot_force = self.net_contact_forces[:, self.feet_indices[0]]
        right_foot_force = self.net_contact_forces[:, self.feet_indices[1]]
        
        # Calculate force magnitude difference
        force_diff = torch.norm(left_foot_force, dim=1) - torch.norm(right_foot_force, dim=1)
        
        # Reward for balanced contact forces
        return torch.exp(-torch.abs(force_diff) / 50.0)
    
    def _reward_foot_slip(self):
        """Calculates the reward for minimizing foot slip. The reward is based on the contact forces
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)
