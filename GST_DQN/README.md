# Tableturf Strategy DQN

`GST_DQN` 是一个独立的 DQN 工程目录，用于训练 Tableturf 单智能体策略。

## 目录说明

- `rl_env.py`: 本目录下的单智能体环境封装
- `networks.py`: DQN 状态编码器与动态动作 Q 网络
- `dqn_trainer.py`: 经验回放、target network、训练日志与 checkpoint
- `train.py`: 命令行训练入口
- `checkpoints/`: 默认模型输出目录

## 依赖

- Python 3.10+
- `numpy`
- `torch`

## 训练示例

在仓库根目录 `Splat3Tableturf-RL` 下运行：

```bash
python -m GST_DQN.train --map-id Square --p1-deck "player:正方广场" --p2-deck MiniGame_Aori --bot-style aggressive --bot-level high
```

查看帮助：

```bash
python -m GST_DQN.train --help
```

列出地图：

```bash
python -m GST_DQN.train --list-maps
```
