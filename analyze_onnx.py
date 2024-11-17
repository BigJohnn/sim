import onnxruntime as ort
import numpy as np

# 加载 ONNX 模型
model_path = "sim/genesis/logs/zeroth-walking/model_100.onnx"
session = ort.InferenceSession(model_path)

# 获取输入信息
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"Input name: {input_name}, shape: {input_shape}")

# 创建随机输入数据
obs_dim = input_shape[1]  # 获取观测维度
dummy_input = np.random.randn(1, obs_dim).astype(np.float32)

print(f"Dummy input shape: {dummy_input.shape}")
print(f"Dummy input: {dummy_input}")

# 运行模型
outputs = session.run(None, {input_name: dummy_input})
actions = outputs[0]

# 分析输出
print(f"Output actions shape: {actions.shape}")
print(f"First 5 actions: {actions[0][:5]}")
print(f"Actions mean: {np.mean(actions)}")
print(f"Actions std: {np.std(actions)}")
