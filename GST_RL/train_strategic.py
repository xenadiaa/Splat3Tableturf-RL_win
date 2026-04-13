from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

import torch

if __package__:
    from .rl_env import list_map_ids
    from .strategic_env import STYLE_MAP, StrategicTableturfEnv, level_npc_profiles_for_map, map_name_by_id, player_map_deck_selector
    from .strategic_ppo_trainer import StrategicPPOConfig, StrategicPPOTrainer
else:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from rl_env import list_map_ids
    from strategic_env import STYLE_MAP, StrategicTableturfEnv, level_npc_profiles_for_map, map_name_by_id, player_map_deck_selector
    from strategic_ppo_trainer import StrategicPPOConfig, StrategicPPOTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a separate strategic PPO model with dynamic shaping rewards.")
    parser.add_argument("--map-id", type=str, default="Square")
    parser.add_argument("--p1-deck", type=str, default=None)
    parser.add_argument("--p2-deck", type=str, default=None)
    parser.add_argument("--bot-style", type=str, default="balanced", choices=["balanced", "aggressive", "conservative"])
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
        default=str(Path(__file__).resolve().parent / "strategy_tiers" / "low_granularity" / "checkpoints" / "strategic_reward"),
    )
    parser.add_argument("--save-every-updates", type=int, default=10)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--list-maps", action="store_true")
    parser.add_argument("--train-all-maps-vs-lv3-npc", action="store_true")
    return parser


def _greedy_eval(trainer: StrategicPPOTrainer, env: StrategicTableturfEnv, episodes: int) -> Dict[str, float]:
    wins = 0
    draws = 0
    losses = 0
    device = trainer.device
    model = trainer.model
    model.eval()
    for _ in range(episodes):
        map_obs, scalar_obs, action_feats = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                logits, _ = model.forward_single(
                    torch.as_tensor(map_obs, dtype=torch.float32, device=device),
                    torch.as_tensor(scalar_obs, dtype=torch.float32, device=device),
                    torch.as_tensor(action_feats, dtype=torch.float32, device=device),
                )
                action_idx = int(torch.argmax(logits).item())
            step = env.step(action_idx)
            map_obs, scalar_obs, action_feats, done = step.map_obs, step.scalar_obs, step.action_features, step.done
        if env.state is not None and env.state.winner == "P1":
            wins += 1
        elif env.state is not None and env.state.winner == "draw":
            draws += 1
        else:
            losses += 1
    n = float(max(1, episodes))
    return {"win_rate": wins / n, "draw_rate": draws / n, "loss_rate": losses / n}


def main() -> None:
    args = build_parser().parse_args()

    if args.list_maps:
        for map_id in list_map_ids():
            print(map_id)
        return

    cfg = StrategicPPOConfig(
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

    if args.train_all_maps_vs_lv3_npc:
        results = []
        for i, map_id in enumerate(list_map_ids()):
            npc_profiles = level_npc_profiles_for_map(level=2, map_id=map_id)
            if not npc_profiles:
                print(f"[skip] map={map_id} reason=no_lv3_npc_profile")
                continue
            npc = npc_profiles[0]
            env = StrategicTableturfEnv(
                map_id=map_id,
                p1_deck_selector=player_map_deck_selector(map_id),
                p2_deck_selector=npc["p2_deck"],
                bot_style=STYLE_MAP[npc["bot_style_raw"]],
                bot_level="high",
                seed=args.seed + i,
            )
            save_dir = str(Path(args.save_dir) / map_id)
            print(
                f"[strategic-train-all] map={map_id} map_name={map_name_by_id(map_id)} "
                f"npc={npc['npc_rowid']} p1={env.p1_deck_rowid} p2={env.p2_deck_rowid} save_dir={save_dir}"
            )
            trainer = StrategicPPOTrainer(
                env=env,
                config=cfg,
                device=args.device,
                save_dir=save_dir,
                run_name=f"strategic_{map_id}_vs_{npc['npc_rowid']}",
                run_config={
                    "mode": "strategic_train_all_maps_vs_lv3_npc",
                    "map_id": map_id,
                    "map_name": map_name_by_id(map_id),
                    "npc_rowid": npc["npc_rowid"],
                    "npc_name": npc["npc_name"],
                    "npc_level": 2,
                    "p1_deck": env.p1_deck_rowid,
                    "p2_deck": env.p2_deck_rowid,
                    "bot_style": STYLE_MAP[npc["bot_style_raw"]],
                    "bot_level": "high",
                    "seed": args.seed + i,
                },
            )
            if args.resume_checkpoint:
                trainer.load_checkpoint(args.resume_checkpoint)
            trainer.train()
            eval_result = _greedy_eval(trainer, env, episodes=30)
            result = {
                "map_id": map_id,
                "map_name": map_name_by_id(map_id),
                "npc_rowid": npc["npc_rowid"],
                "npc_name": npc["npc_name"],
                "p1_deck": env.p1_deck_rowid,
                "p2_deck": env.p2_deck_rowid,
                **eval_result,
                "save_dir": save_dir,
            }
            results.append(result)
            Path(save_dir, "eval_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = {"mode": "strategic_train_all_maps_vs_lv3_npc", "results": results}
        summary_path = Path(args.save_dir) / "batch_eval_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[strategic-batch-summary] path={summary_path}")
        return

    env = StrategicTableturfEnv(
        map_id=args.map_id,
        p1_deck_selector=args.p1_deck,
        p2_deck_selector=args.p2_deck,
        bot_style=args.bot_style,
        bot_level=args.bot_level,
        seed=args.seed,
    )
    print(f"[strategic-train] map={args.map_id} p1_deck={env.p1_deck_rowid} p2_deck={env.p2_deck_rowid}")
    trainer = StrategicPPOTrainer(
        env=env,
        config=cfg,
        device=args.device,
        save_dir=args.save_dir,
        run_name=f"strategic_{args.map_id}",
        run_config={
            "map_id": args.map_id,
            "p1_deck": env.p1_deck_rowid,
            "p2_deck": env.p2_deck_rowid,
            "bot_style": args.bot_style,
            "bot_level": args.bot_level,
            "seed": args.seed,
            "reward_mode": "situation_dependent_shaping",
        },
    )
    if args.resume_checkpoint:
        trainer.load_checkpoint(args.resume_checkpoint)
    trainer.train()


if __name__ == "__main__":
    main()
