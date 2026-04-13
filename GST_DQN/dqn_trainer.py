from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import random
import time
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import torch
from torch import nn

if __package__:
    from .networks import DQNNet
    from .rl_env import TableturfRLEnv
else:
    from networks import DQNNet
    from rl_env import TableturfRLEnv


@dataclass
class DQNConfig:
    total_steps: int = 100_000
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 50_000
    warmup_steps: int = 2_000
    train_freq: int = 4
    target_update_freq: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    max_grad_norm: float = 1.0
    save_every_steps: int = 10_000
    log_every_steps: int = 1_000


@dataclass
class Transition:
    map_obs: np.ndarray
    scalar_obs: np.ndarray
    action_features: np.ndarray
    action_index: int
    reward: float
    next_map_obs: np.ndarray
    next_scalar_obs: np.ndarray
    next_action_features: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(list(self._buffer), batch_size)


class DQNTrainer:
    def __init__(
        self,
        env: TableturfRLEnv,
        config: DQNConfig,
        device: str = "cpu",
        save_dir: str = "checkpoints",
        run_name: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.env = env
        self.cfg = config
        self.device = torch.device(device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or "train_dqn"
        self.run_config = run_config or {}
        self.metrics_jsonl_path = self.save_dir / "training_metrics.jsonl"
        self.metrics_csv_path = self.save_dir / "training_metrics.csv"
        self.summary_json_path = self.save_dir / "training_summary.json"

        self.model = DQNNet(
            map_channels=env.map_shape[0],
            scalar_dim=env.scalar_dim,
            action_feature_dim=env.action_feature_dim,
        ).to(self.device)
        self.target_model = DQNNet(
            map_channels=env.map_shape[0],
            scalar_dim=env.scalar_dim,
            action_feature_dim=env.action_feature_dim,
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        self.replay = ReplayBuffer(self.cfg.replay_capacity)
        self._write_meta()

        self._map_obs, self._scalar_obs, self._action_feats = self.env.reset()
        self._ep_reward = 0.0
        self._recent_ep_rewards: List[float] = []
        self._recent_wins: List[float] = []
        self._loss_history: List[float] = []

    def train(self) -> None:
        start_t = time.time()

        for step in range(1, self.cfg.total_steps + 1):
            epsilon = self._epsilon(step)
            action_idx = self._select_action(self._map_obs, self._scalar_obs, self._action_feats, epsilon)
            env_step = self.env.step(action_idx)

            self.replay.add(
                Transition(
                    map_obs=self._map_obs.copy(),
                    scalar_obs=self._scalar_obs.copy(),
                    action_features=self._action_feats.copy(),
                    action_index=action_idx,
                    reward=env_step.reward,
                    next_map_obs=env_step.map_obs.copy(),
                    next_scalar_obs=env_step.scalar_obs.copy(),
                    next_action_features=env_step.action_features.copy(),
                    done=env_step.done,
                )
            )

            self._ep_reward += env_step.reward
            self._map_obs, self._scalar_obs, self._action_feats = (
                env_step.map_obs,
                env_step.scalar_obs,
                env_step.action_features,
            )

            if env_step.done:
                self._recent_ep_rewards.append(self._ep_reward)
                winner = env_step.info.get("winner", "")
                self._recent_wins.append(1.0 if winner == "P1" else 0.0)
                self._ep_reward = 0.0
                self._map_obs, self._scalar_obs, self._action_feats = self.env.reset()

            if step >= self.cfg.warmup_steps and step % self.cfg.train_freq == 0 and len(self.replay) >= self.cfg.batch_size:
                self._loss_history.append(self._train_batch())

            if step % self.cfg.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            if step % self.cfg.save_every_steps == 0 or step == self.cfg.total_steps:
                ckpt = self.save_dir / f"dqn_tableturf_s{step:06d}.pt"
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "target_model": self.target_model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    ckpt,
                )

            if step % self.cfg.log_every_steps == 0 or step == self.cfg.total_steps:
                elapsed = time.time() - start_t
                mean_r = float(np.mean(self._recent_ep_rewards[-20:])) if self._recent_ep_rewards else 0.0
                win_rate = float(np.mean(self._recent_wins[-50:])) if self._recent_wins else 0.0
                mean_loss = float(np.mean(self._loss_history[-100:])) if self._loss_history else 0.0
                self._append_update_metric(step, mean_loss, epsilon, len(self.replay), mean_r, win_rate, elapsed)
                print(
                    f"[step {step}/{self.cfg.total_steps}] "
                    f"loss={mean_loss:.4f} epsilon={epsilon:.3f} replay={len(self.replay)} "
                    f"mean_ep_reward={mean_r:.3f} win_rate={win_rate:.3f} elapsed={elapsed:.1f}s"
                )

        self._write_summary(total_elapsed=time.time() - start_t)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        obj = torch.load(checkpoint_path, map_location=self.device)
        state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
        self.model.load_state_dict(state_dict)
        if isinstance(obj, dict) and "target_model" in obj:
            self.target_model.load_state_dict(obj["target_model"])
        else:
            self.target_model.load_state_dict(state_dict)
        if isinstance(obj, dict) and "optimizer" in obj:
            self.optimizer.load_state_dict(obj["optimizer"])

    def _epsilon(self, step: int) -> float:
        if self.cfg.epsilon_decay_steps <= 0:
            return self.cfg.epsilon_end
        frac = min(1.0, step / float(self.cfg.epsilon_decay_steps))
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def _select_action(
        self,
        map_obs: np.ndarray,
        scalar_obs: np.ndarray,
        action_features: np.ndarray,
        epsilon: float,
    ) -> int:
        if random.random() < epsilon:
            return random.randrange(action_features.shape[0])

        map_t = torch.as_tensor(map_obs, dtype=torch.float32, device=self.device)
        scalar_t = torch.as_tensor(scalar_obs, dtype=torch.float32, device=self.device)
        action_feat_t = torch.as_tensor(action_features, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.model.forward_single(map_t, scalar_t, action_feat_t)
        return int(torch.argmax(q_values).item())

    def _train_batch(self) -> float:
        batch = self.replay.sample(self.cfg.batch_size)
        losses = []

        for transition in batch:
            map_t = torch.as_tensor(transition.map_obs, dtype=torch.float32, device=self.device)
            scalar_t = torch.as_tensor(transition.scalar_obs, dtype=torch.float32, device=self.device)
            action_feat_t = torch.as_tensor(transition.action_features, dtype=torch.float32, device=self.device)

            q_values = self.model.forward_single(map_t, scalar_t, action_feat_t)
            q_selected = q_values[transition.action_index]

            reward_t = torch.tensor(transition.reward, dtype=torch.float32, device=self.device)
            done_t = torch.tensor(float(transition.done), dtype=torch.float32, device=self.device)

            with torch.no_grad():
                if transition.done:
                    next_q = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                else:
                    next_map_t = torch.as_tensor(transition.next_map_obs, dtype=torch.float32, device=self.device)
                    next_scalar_t = torch.as_tensor(transition.next_scalar_obs, dtype=torch.float32, device=self.device)
                    next_action_feat_t = torch.as_tensor(
                        transition.next_action_features,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    next_q = self.target_model.forward_single(next_map_t, next_scalar_t, next_action_feat_t).max()
                target = reward_t + self.cfg.gamma * next_q * (1.0 - done_t)

            losses.append((q_selected - target).pow(2))

        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())

    def _write_meta(self) -> None:
        self._append_jsonl(
            {
                "event": "meta",
                "utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "run_name": self.run_name,
                "save_dir": str(self.save_dir),
                "device": str(self.device),
                "config": {
                    "total_steps": self.cfg.total_steps,
                    "gamma": self.cfg.gamma,
                    "learning_rate": self.cfg.learning_rate,
                    "batch_size": self.cfg.batch_size,
                    "replay_capacity": self.cfg.replay_capacity,
                    "warmup_steps": self.cfg.warmup_steps,
                    "train_freq": self.cfg.train_freq,
                    "target_update_freq": self.cfg.target_update_freq,
                    "epsilon_start": self.cfg.epsilon_start,
                    "epsilon_end": self.cfg.epsilon_end,
                    "epsilon_decay_steps": self.cfg.epsilon_decay_steps,
                    "max_grad_norm": self.cfg.max_grad_norm,
                    "save_every_steps": self.cfg.save_every_steps,
                    "log_every_steps": self.cfg.log_every_steps,
                },
                "run_config": self.run_config,
            }
        )

    def _append_update_metric(
        self,
        step: int,
        loss: float,
        epsilon: float,
        replay_size: int,
        mean_ep_reward: float,
        win_rate: float,
        elapsed: float,
    ) -> None:
        row = {
            "event": "update",
            "utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "step": step,
            "loss": loss,
            "epsilon": epsilon,
            "replay_size": replay_size,
            "mean_ep_reward": mean_ep_reward,
            "win_rate": win_rate,
            "elapsed_sec": elapsed,
        }
        self._append_jsonl(row)
        self._append_csv(row)

    def _write_summary(self, total_elapsed: float) -> None:
        summary = {
            "utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "run_name": self.run_name,
            "total_steps_seen": self.cfg.total_steps,
            "total_elapsed_sec": total_elapsed,
            "final_recent_mean_ep_reward": float(np.mean(self._recent_ep_rewards[-20:])) if self._recent_ep_rewards else 0.0,
            "final_recent_win_rate": float(np.mean(self._recent_wins[-50:])) if self._recent_wins else 0.0,
            "final_recent_loss": float(np.mean(self._loss_history[-100:])) if self._loss_history else 0.0,
            "episodes_finished": len(self._recent_ep_rewards),
            "save_dir": str(self.save_dir),
            "run_config": self.run_config,
        }
        self.summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self._append_jsonl({"event": "summary", **summary})

    def _append_jsonl(self, obj: Dict[str, Any]) -> None:
        with self.metrics_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _append_csv(self, row: Dict[str, Any]) -> None:
        fieldnames = [
            "event",
            "utc",
            "step",
            "loss",
            "epsilon",
            "replay_size",
            "mean_ep_reward",
            "win_rate",
            "elapsed_sec",
        ]
        write_header = not self.metrics_csv_path.exists()
        with self.metrics_csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fieldnames})
