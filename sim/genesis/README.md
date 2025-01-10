# 进入genesis目录。
`cd sim/genesis` 

# 安装rsl_rl。
```
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl && git checkout v1.0.2 && pip install -e .
```

# 安装tensorboard。
`pip install tensorboard`

安装完成后，通过运行以下命令开始训练：

`python zeroth_train.py`

要监控训练过程，请启动TensorBoard：

`tensorboard --logdir logs`

查看训练结果。

`python zeroth_eval.py`