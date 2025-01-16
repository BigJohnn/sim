import torch
import torch.nn as nn
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Convert model to ONNX format')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to config file (.pkl)')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model file (.pt)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output ONNX file path')
    return parser.parse_args()

args = parse_args()

# 加载配置
with open(args.cfg, 'rb') as f:
    cfgs = pickle.load(f)

# 加载模型
model_dict = torch.load(args.model, weights_only=True)

# 创建继承自ActorCritic的模型类
from rsl_rl.modules import ActorCritic

class ExportModel(ActorCritic):
    def forward(self, obs):
        # 使用actor网络生成动作
        actions = self.actor(obs)
        # 使用critic网络评估状态值
        values = self.critic(obs)
        return actions, values

# 根据配置创建模型实例        
model = ExportModel(
    num_actor_obs=cfgs[1]['num_obs'],
    num_critic_obs=cfgs[1]['num_obs'],
    num_actions=cfgs[0]['num_actions'],
    actor_hidden_dims=cfgs[4]['policy']['actor_hidden_dims'],
    critic_hidden_dims=cfgs[4]['policy']['critic_hidden_dims'],
    activation='elu',
    init_noise_std=cfgs[4]['policy']['init_noise_std']
)

# 加载模型参数
model.load_state_dict(model_dict['model_state_dict'])
model.eval()

# 创建示例输入
obs_dim = cfgs[1]['num_obs']  # 从obs_cfg获取观测维度
dummy_input = torch.randn(1, obs_dim)

# 转换为ONNX
torch.onnx.export(
    model,
    dummy_input,
    args.output,
    input_names=['obs'],
    output_names=['actions'],
    dynamic_axes={
        'obs': {0: 'batch_size'},
        'actions': {0: 'batch_size'}
    }
)

print("Model successfully converted to ONNX format")
