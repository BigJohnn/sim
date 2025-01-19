# 进入genesis目录。
`cd sim/genesis` 

# 安装rsl_rl。
```
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl && git checkout v2.0.1 && pip install -e .
```

# 安装tensorboard。
`pip install tensorboard`

安装完成后，通过运行以下命令开始训练：

`python zeroth_train.py`

要监控训练过程，请启动TensorBoard：

`tensorboard --logdir logs`

查看训练结果。

`python zeroth_eval.py`

模型转换。

`cd genesis`

`python3 utils/convert_to_onnx.py --cfg logs/zeroth-walking/cfgs.pkl --model ../../examples/model_100.pt --output ../../examples/model_100.onnx`

注意⚠️
Mac M系列芯片，使用micromamba替换conda使用，例如
`micromamba activate genesis`
然后再调用pip/python等命令