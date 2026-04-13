from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__:
    from .selfplay_env import TableturfSelfPlayEnv
    from .selfplay_ppo_trainer import SelfPlayPPOConfig, SelfPlayPPOTrainer
else:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from selfplay_env import TableturfSelfPlayEnv
    from selfplay_ppo_trainer import SelfPlayPPOConfig, SelfPlayPPOTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a 2P shared-policy PPO model.")
    parser.add_argument("--map-id", type=str, required=True)
    parser.add_argument("--p1-deck", type=str, required=True)
    parser.add_argument("--p2-deck", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--total-steps", type=int, default=8192)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--save-every-updates", type=int, default=8)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "strategy_tiers" / "low_granularity" / "checkpoints" / "selfplay_2p"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = SelfPlayPPOConfig(
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
    env = TableturfSelfPlayEnv(
        map_id=args.map_id,
        p1_deck_selector=args.p1_deck,
        p2_deck_selector=args.p2_deck,
        seed=args.seed,
    )
    print(f"[selfplay-train] map={args.map_id} p1_deck={args.p1_deck} p2_deck={args.p2_deck}")
    trainer = SelfPlayPPOTrainer(
        env=env,
        config=cfg,
        device=args.device,
        save_dir=args.save_dir,
        run_name=f"selfplay_{args.map_id}",
        run_config={
            "map_id": args.map_id,
            "p1_deck": args.p1_deck,
            "p2_deck": args.p2_deck,
            "seed": args.seed,
            "mode": "2P_shared_policy",
        },
    )
    trainer.train()


if __name__ == "__main__":
    main()
