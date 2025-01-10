# 跑在mac m系列芯片上的sim软件，
# 安装micromamba, python3.10
# micromamba activate gensis
# python sim_mac_mx.py
import genesis as gs
import numpy as np

gs.init(backend=gs.metal)

scene = gs.Scene(show_viewer=True,
                 viewer_options = gs.options.ViewerOptions(
        res           = (1280, 960),
        camera_pos    = (3.5, 0.0, 0.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = True, # 显示原点坐标系
        world_frame_size = 1.0, # 坐标系长度(米)
        show_link_frame  = False, # 不显示实体链接坐标系 
        show_cameras     = False, # 不显示相机网格和视锥
        plane_reflection = True, # 开启平面反射
        ambient_light    = (0.1, 0.1, 0.1), # 环境光
    ),
    renderer = gs.renderers.Rasterizer(), # 使用光栅化渲染器
    )

plane = scene.add_entity(gs.morphs.Plane())
zeroth01 = scene.add_entity(
    gs.morphs.URDF(file='./sim/resources/stompymicro/robot.urdf', pos=(0,0,0.3), fixed=True),
)

scene.build()

jnt_names = [
    'right_ankle_pitch',
    'right_knee_pitch',
    'right_hip_roll',
    'right_hip_yaw',
    'right_hip_pitch',
    'left_ankle_pitch',
    'left_knee_pitch',
    'left_hip_roll',
    'left_hip_yaw',
    'left_hip_pitch',
    'righ_elbow_yaw',
    'right_shoulder_yaw',
    'right_shoulder_pitch',
    'left_shoulder_pitch',
    'left_shoulder_yaw',
    'left_elbow_yaw'
]

dofs_idx = [zeroth01.get_joint(name).dof_idx_local for name in jnt_names]

# 设置初始位置为limits的中值
init_positions = np.array([
    0.0,  # right_ankle_pitch
    0.0,  # right_knee_pitch
    0.0,  # right_hip_roll
    0.0,  # right_hip_yaw
    0.0,  # right_hip_pitch
    0.0,  # left_ankle_pitch
    0.0,  # left_knee_pitch
    0.0,  # left_hip_roll
    0.0,  # left_hip_yaw
    0.0,  # left_hip_pitch
    3.14, # righ_elbow_yaw
    0.0,  # right_shoulder_yaw
    0.0,  # right_shoulder_pitch
    0.0,  # left_shoulder_pitch
    0.0,  # left_shoulder_yaw
    3.14  # left_elbow_yaw
])

zeroth01.control_dofs_position(init_positions, dofs_idx)

def run_sim(scene,):
    while True:
        scene.step()
        
gs.tools.run_in_another_thread(fn=run_sim, args=(scene,)) # start the simulation in another thread
scene.viewer.start() # start the viewer in the main thread (the render thread)
