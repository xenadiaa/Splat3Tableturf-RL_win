# Tableturf Strategy RL

This folder provides a PPO training pipeline for learning Tableturf strategy with `tableturf_sim`.

## Files

- `rl_env.py`: Single-agent wrapper over `tableturf_sim` (`P1` is RL agent, `P2` is built-in bot).
- `networks.py`: Dynamic-action policy/value neural network.
- `ppo_trainer.py`: PPO rollout + update logic.
- `train.py`: CLI entry.

## Requirements

- Python 3.10+
- `numpy`
- `torch`

Install example:

```bash
pip install numpy torch
```

## Train

From repo root (`Splat3Tableturf-RL`):

```bash
python -m GST_RL.train --total-steps 100000 --map-id ManySp --bot-style balanced --bot-level mid
```

List map IDs:

```bash
python -m GST_RL.train --list-maps
```

List preset deck row IDs:

```bash
python -m GST_RL.train --list-decks
```

Use a specific deck by row id or numeric index:

```bash
python -m GST_RL.train --p1-deck MiniGame_Aori --p2-deck 3
```

Auto deck selection by map (derived from NPC map/deck relations):

```bash
python -m GST_RL.train --list-map-decks
```

Train all maps (one model directory per map):

```bash
python -m GST_RL.train --train-all-maps --total-steps 50000 --save-dir GST_RL/checkpoints_all_maps
```

Resume from checkpoint:

```bash
python -m GST_RL.train --map-id Cross --resume-checkpoint GST_RL/checkpoints/ppo_tableturf_u0010.pt
```

Training logs are written into each `save-dir`:

- `training_metrics.jsonl`: per-update records (loss/reward/win_rate/elapsed)
- `training_metrics.csv`: per-update table
- `training_summary.json`: final summary

Checkpoints are written to `GST_RL/checkpoints/`.

## Strategic PPO

This repo also includes a separate PPO pipeline with a richer, situation-dependent shaping reward:

- `strategic_env.py`: strategic reward wrapper with breakthrough/compression/SP/fortify shaping
- `strategic_networks.py`: separate policy/value network for strategic reward training
- `strategic_ppo_trainer.py`: separate PPO trainer and logs
- `train_strategic.py`: CLI entry

Example:

```bash
python -m GST_RL.train_strategic --map-id Square --p1-deck "player:正方广场" --p2-deck MiniGame_Aori --bot-style aggressive --bot-level high
```

## 2P Self-Play PPO

Shared-policy 2P PPO training is available separately:

- `selfplay_env.py`
- `selfplay_ppo_trainer.py`
- `train_selfplay.py`

Example:

```bash
python -m GST_RL.train_selfplay --map-id Square --p1-deck "player:正方广场" --p2-deck MiniGame_Aori
```

Dual-policy 2P PPO is also available:

```bash
python -m GST_RL.train_selfplay_dual --map-id Square --p1-deck "player:正方广场" --p2-deck MiniGame_Aori
```

Score-first reward PPO is available separately:

```bash
python -m GST_RL.train_score_reward --map-id Square --p1-deck "player:正方广场" --p2-deck MiniGame_Aori --bot-style aggressive --bot-level high
```

Aggressive-opening + basic PPO is available separately:

```bash
python -m GST_RL.train_opening_aggressive --map-id Square --p1-deck "player:正方广场" --p2-deck MiniGame_Aori --opening-turns 3 --bot-style aggressive --bot-level high
```
