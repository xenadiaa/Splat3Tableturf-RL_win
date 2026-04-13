from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

if __package__:
    from .networks import PolicyValueNet
    from .selfplay_env import TableturfSelfPlayEnv
else:
    from networks import PolicyValueNet
    from selfplay_env import TableturfSelfPlayEnv


@dataclass
class DualSelfPlayPPOConfig:
    total_steps: int = 16_384
    rollout_steps: int = 256
    epochs: int = 2
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    save_every_updates: int = 8


@dataclass
class DualSelfPlayStep:
    map_obs: np.ndarray
    scalar_obs: np.ndarray
    action_features: np.ndarray
    action_index: int
    log_prob: float
    value: float
    reward: float
    done: bool


class DualSelfPlayPPOTrainer:
    def __init__(
        self,
        env: TableturfSelfPlayEnv,
        config: DualSelfPlayPPOConfig,
        device: str = "cpu",
        save_dir: str = "checkpoints_selfplay_dual",
        run_name: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.env = env
        self.cfg = config
        self.device = torch.device(device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or "train_selfplay_dual"
        self.run_config = run_config or {}
        self.metrics_jsonl_path = self.save_dir / "training_metrics.jsonl"
        self.metrics_csv_path = self.save_dir / "training_metrics.csv"
        self.summary_json_path = self.save_dir / "training_summary.json"

        self.p1_model = PolicyValueNet(
            map_channels=env.map_shape[0],
            scalar_dim=env.scalar_dim,
            action_feature_dim=env.action_feature_dim,
        ).to(self.device)
        self.p2_model = PolicyValueNet(
            map_channels=env.map_shape[0],
            scalar_dim=env.scalar_dim,
            action_feature_dim=env.action_feature_dim,
        ).to(self.device)
        self.p1_optimizer = torch.optim.Adam(self.p1_model.parameters(), lr=self.cfg.learning_rate)
        self.p2_optimizer = torch.optim.Adam(self.p2_model.parameters(), lr=self.cfg.learning_rate)
        self._recent_ep_rewards: List[float] = []
        self._recent_p1_wins: List[float] = []
        self._write_meta()

    def train(self) -> None:
        total_updates = max(1, self.cfg.total_steps // self.cfg.rollout_steps)
        start_t = time.time()
        for update in range(1, total_updates + 1):
            p1_steps, p2_steps, p1_last_value, p2_last_value = self.collect_rollout(self.cfg.rollout_steps)
            p1_returns, p1_advantages = self.compute_gae(p1_steps, p1_last_value)
            p2_returns, p2_advantages = self.compute_gae(p2_steps, p2_last_value)
            p1_losses = self.ppo_update(self.p1_model, self.p1_optimizer, p1_steps, p1_returns, p1_advantages)
            p2_losses = self.ppo_update(self.p2_model, self.p2_optimizer, p2_steps, p2_returns, p2_advantages)

            if update % self.cfg.save_every_updates == 0 or update == total_updates:
                torch.save(
                    {"model": self.p1_model.state_dict(), "optimizer": self.p1_optimizer.state_dict()},
                    self.save_dir / f"p1_ppo_u{update:04d}.pt",
                )
                torch.save(
                    {"model": self.p2_model.state_dict(), "optimizer": self.p2_optimizer.state_dict()},
                    self.save_dir / f"p2_ppo_u{update:04d}.pt",
                )

            elapsed = time.time() - start_t
            mean_r = float(np.mean(self._recent_ep_rewards[-20:])) if self._recent_ep_rewards else 0.0
            win_rate = float(np.mean(self._recent_p1_wins[-50:])) if self._recent_p1_wins else 0.0
            self._append_update_metric(update, total_updates, p1_losses, p2_losses, mean_r, win_rate, elapsed)
            print(
                f"[dual-selfplay update {update}/{total_updates}] "
                f"p1_policy={p1_losses['policy']:.4f} p2_policy={p2_losses['policy']:.4f} "
                f"mean_turn_reward={mean_r:.3f} p1_win_rate={win_rate:.3f} elapsed={elapsed:.1f}s"
            )
        self._write_summary(total_updates, time.time() - start_t)

    def collect_rollout(self, n_steps_per_player: int) -> Tuple[List[DualSelfPlayStep], List[DualSelfPlayStep], float, float]:
        p1_steps: List[DualSelfPlayStep] = []
        p2_steps: List[DualSelfPlayStep] = []
        self.env.reset()
        ep_turn_reward = 0.0

        while len(p1_steps) < n_steps_per_player or len(p2_steps) < n_steps_per_player:
            prev_diff = self.env.score_diff()
            pending: Dict[str, Dict[str, object]] = {}
            for player, model in (("P1", self.p1_model), ("P2", self.p2_model)):
                map_obs, scalar_obs, action_feats, actions = self.env.observe(player)
                with torch.no_grad():
                    logits, value = model.forward_single(
                        torch.as_tensor(map_obs, dtype=torch.float32, device=self.device),
                        torch.as_tensor(scalar_obs, dtype=torch.float32, device=self.device),
                        torch.as_tensor(action_feats, dtype=torch.float32, device=self.device),
                    )
                    dist = Categorical(logits=logits)
                    action_idx = int(dist.sample().item())
                    log_prob = float(dist.log_prob(torch.tensor(action_idx, device=self.device)).item())
                    value_f = float(value.item())
                pending[player] = {
                    "map_obs": map_obs,
                    "scalar_obs": scalar_obs,
                    "action_features": action_feats,
                    "action_index": action_idx,
                    "log_prob": log_prob,
                    "value": value_f,
                    "action": actions[action_idx],
                }
                result = self.env.submit(actions[action_idx], prev_diff)
                if player == "P2":
                    done = result.done
                    if len(p1_steps) < n_steps_per_player:
                        p1_steps.append(
                            DualSelfPlayStep(
                                map_obs=pending["P1"]["map_obs"],  # type: ignore[arg-type]
                                scalar_obs=pending["P1"]["scalar_obs"],  # type: ignore[arg-type]
                                action_features=pending["P1"]["action_features"],  # type: ignore[arg-type]
                                action_index=pending["P1"]["action_index"],  # type: ignore[arg-type]
                                log_prob=pending["P1"]["log_prob"],  # type: ignore[arg-type]
                                value=pending["P1"]["value"],  # type: ignore[arg-type]
                                reward=result.reward_p1,
                                done=done,
                            )
                        )
                    if len(p2_steps) < n_steps_per_player:
                        p2_steps.append(
                            DualSelfPlayStep(
                                map_obs=pending["P2"]["map_obs"],  # type: ignore[arg-type]
                                scalar_obs=pending["P2"]["scalar_obs"],  # type: ignore[arg-type]
                                action_features=pending["P2"]["action_features"],  # type: ignore[arg-type]
                                action_index=pending["P2"]["action_index"],  # type: ignore[arg-type]
                                log_prob=pending["P2"]["log_prob"],  # type: ignore[arg-type]
                                value=pending["P2"]["value"],  # type: ignore[arg-type]
                                reward=result.reward_p2,
                                done=done,
                            )
                        )
                    ep_turn_reward += result.reward_p1
                    if done:
                        self._recent_ep_rewards.append(ep_turn_reward)
                        winner = result.info.get("winner", "")
                        self._recent_p1_wins.append(1.0 if winner == "P1" else 0.0)
                        ep_turn_reward = 0.0
                        self.env.reset()
                    break

        p1_last_value = self._bootstrap_value(self.p1_model, "P1", p1_steps)
        p2_last_value = self._bootstrap_value(self.p2_model, "P2", p2_steps)
        return p1_steps, p2_steps, p1_last_value, p2_last_value

    def _bootstrap_value(self, model: PolicyValueNet, player: str, steps: List[DualSelfPlayStep]) -> float:
        if steps and steps[-1].done:
            return 0.0
        map_obs, scalar_obs, action_feats, _ = self.env.observe(player)
        with torch.no_grad():
            _, value = model.forward_single(
                torch.as_tensor(map_obs, dtype=torch.float32, device=self.device),
                torch.as_tensor(scalar_obs, dtype=torch.float32, device=self.device),
                torch.as_tensor(action_feats, dtype=torch.float32, device=self.device),
            )
            return float(value.item())

    def compute_gae(self, steps: List[DualSelfPlayStep], last_value: float) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.array([s.reward for s in steps], dtype=np.float32)
        values = np.array([s.value for s in steps], dtype=np.float32)
        dones = np.array([s.done for s in steps], dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_adv = 0.0
        next_value = last_value
        for t in reversed(range(len(steps))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_value * nonterminal - values[t]
            last_adv = delta + self.cfg.gamma * self.cfg.gae_lambda * nonterminal * last_adv
            advantages[t] = last_adv
            next_value = values[t]
        returns = advantages + values
        return returns, advantages

    def ppo_update(
        self,
        model: PolicyValueNet,
        optimizer: torch.optim.Optimizer,
        steps: List[DualSelfPlayStep],
        returns: np.ndarray,
        advantages: np.ndarray,
    ) -> Dict[str, float]:
        idxs = np.arange(len(steps))
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        for _ in range(self.cfg.epochs):
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), self.cfg.minibatch_size):
                mb_idx = idxs[start : start + self.cfg.minibatch_size]
                if len(mb_idx) == 0:
                    continue
                mb_policy = []
                mb_value = []
                mb_entropy = []
                for i in mb_idx:
                    s = steps[int(i)]
                    logits, value = model.forward_single(
                        torch.as_tensor(s.map_obs, dtype=torch.float32, device=self.device),
                        torch.as_tensor(s.scalar_obs, dtype=torch.float32, device=self.device),
                        torch.as_tensor(s.action_features, dtype=torch.float32, device=self.device),
                    )
                    dist = Categorical(logits=logits)
                    action_t = torch.tensor(s.action_index, dtype=torch.long, device=self.device)
                    new_log_prob = dist.log_prob(action_t)
                    old_log_prob = torch.tensor(s.log_prob, dtype=torch.float32, device=self.device)
                    ratio = torch.exp(new_log_prob - old_log_prob)
                    adv_t = torch.tensor(float(adv[int(i)]), dtype=torch.float32, device=self.device)
                    pg1 = ratio * adv_t
                    pg2 = torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef) * adv_t
                    mb_policy.append(-torch.min(pg1, pg2))
                    ret_t = torch.tensor(float(returns[int(i)]), dtype=torch.float32, device=self.device)
                    mb_value.append((value - ret_t).pow(2))
                    mb_entropy.append(dist.entropy())
                policy_loss = torch.stack(mb_policy).mean()
                value_loss = torch.stack(mb_value).mean()
                entropy = torch.stack(mb_entropy).mean()
                loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                optimizer.step()
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
        return {
            "policy": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

    def _write_meta(self) -> None:
        self._append_jsonl(
            {
                "event": "meta",
                "utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "run_name": self.run_name,
                "save_dir": str(self.save_dir),
                "device": str(self.device),
                "config": self.cfg.__dict__,
                "run_config": self.run_config,
            }
        )

    def _append_update_metric(
        self,
        update: int,
        total_updates: int,
        p1_losses: Dict[str, float],
        p2_losses: Dict[str, float],
        mean_ep_reward: float,
        win_rate: float,
        elapsed: float,
    ) -> None:
        row = {
            "event": "update",
            "utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "update": update,
            "total_updates": total_updates,
            "steps_seen_per_player": update * self.cfg.rollout_steps,
            "p1_policy_loss": p1_losses["policy"],
            "p1_value_loss": p1_losses["value"],
            "p2_policy_loss": p2_losses["policy"],
            "p2_value_loss": p2_losses["value"],
            "mean_turn_reward": mean_ep_reward,
            "p1_win_rate": win_rate,
            "elapsed_sec": elapsed,
        }
        self._append_jsonl(row)
        self._append_csv(row)

    def _write_summary(self, total_updates: int, total_elapsed: float) -> None:
        summary = {
            "utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "run_name": self.run_name,
            "total_updates": total_updates,
            "total_steps_seen_per_player": total_updates * self.cfg.rollout_steps,
            "total_elapsed_sec": total_elapsed,
            "final_recent_mean_turn_reward": float(np.mean(self._recent_ep_rewards[-20:])) if self._recent_ep_rewards else 0.0,
            "final_recent_p1_win_rate": float(np.mean(self._recent_p1_wins[-50:])) if self._recent_p1_wins else 0.0,
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
            "update",
            "total_updates",
            "steps_seen_per_player",
            "p1_policy_loss",
            "p1_value_loss",
            "p2_policy_loss",
            "p2_value_loss",
            "mean_turn_reward",
            "p1_win_rate",
            "elapsed_sec",
        ]
        write_header = not self.metrics_csv_path.exists()
        with self.metrics_csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fieldnames})
