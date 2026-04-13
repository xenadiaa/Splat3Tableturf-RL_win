from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__:
    from .ppo_trainer import PPOConfig, PPOTrainer
    from .score_reward_env import TableturfScoreRewardEnv
else:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from ppo_trainer import PPOConfig, PPOTrainer
    from score_reward_env import TableturfScoreRewardEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO with score-first reward and SP misuse penalty.")
    parser.add_argument("--map-id", type=str, default="Square")
    parser.add_argument("--p1-deck", type=str, default=None)
    parser.add_argument("--p2-deck", type=str, default=None)
    parser.add_argument("--bot-style", type=str, default="aggressive", choices=["balanced", "aggressive", "conservative"])
    parser.add_argument("--bot-level", type=str, default="high", choices=["low", "mid", "high"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "strategy_tiers" / "low_granularity" / "checkpoints" / "score_reward"),
    )
    parser.add_argument("--save-every-updates", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = PPOConfig(
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        save_every_updates=args.save_every_updates,
    )
    env = TableturfScoreRewardEnv(
        map_id=args.map_id,
        p1_deck_selector=args.p1_deck,
        p2_deck_selector=args.p2_deck,
        bot_style=args.bot_style,
        bot_level=args.bot_level,
        seed=args.seed,
    )
    print(f"[score-reward-train] map={args.map_id} p1_deck={env.p1_deck_rowid} p2_deck={env.p2_deck_rowid}")
    trainer = PPOTrainer(
        env=env,
        config=cfg,
        device=args.device,
        save_dir=args.save_dir,
        run_name=f"score_reward_{args.map_id}",
        run_config={
            "map_id": args.map_id,
            "p1_deck": env.p1_deck_rowid,
            "p2_deck": env.p2_deck_rowid,
            "bot_style": args.bot_style,
            "bot_level": args.bot_level,
            "seed": args.seed,
            "reward_mode": "score_0.7_diff_0.25_win_0.05_with_sp_misuse_penalty",
        },
    )
    trainer.train()


if __name__ == "__main__":
    main()
