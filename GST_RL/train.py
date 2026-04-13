from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

if __package__:
    from .npc_eval import run_eval
    from .rl_env import (
        auto_deck_rowids_for_map,
        level_npc_profiles_for_map,
        list_deck_rowids,
        list_map_ids,
        map_deck_candidates,
        map_name_by_id,
        player_map_deck_selector,
    )
else:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from npc_eval import run_eval
    from rl_env import (
        auto_deck_rowids_for_map,
        level_npc_profiles_for_map,
        list_deck_rowids,
        list_map_ids,
        map_deck_candidates,
        map_name_by_id,
        player_map_deck_selector,
    )


STYLE_MAP = {
    "Aggressive": "aggressive",
    "Balance": "balanced",
    "AccumulateSpecial": "conservative",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Tableturf strategy model with PPO.")
    parser.add_argument("--map-id", type=str, default="ManySp", help="Map id from MiniGameMapInfo.json")
    parser.add_argument("--p1-deck", type=str, default=None, help="Deck rowid or numeric index for agent deck")
    parser.add_argument("--p2-deck", type=str, default=None, help="Deck rowid or numeric index for bot deck")
    parser.add_argument("--bot-style", type=str, default="balanced", choices=["balanced", "aggressive", "conservative"])
    parser.add_argument("--bot-level", type=str, default="mid", choices=["low", "mid", "high"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")

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

    parser.add_argument("--save-dir", type=str, default=str(Path(__file__).resolve().parent / "checkpoints"))
    parser.add_argument("--save-every-updates", type=int, default=10)
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Resume training from checkpoint file")

    parser.add_argument("--list-maps", action="store_true", help="Print available map IDs and exit")
    parser.add_argument("--list-decks", action="store_true", help="Print available preset deck row IDs and exit")
    parser.add_argument("--list-map-decks", action="store_true", help="Print auto-selected decks per map and exit")
    parser.add_argument("--train-all-maps", action="store_true", help="Train one model per map")
    parser.add_argument(
        "--train-all-maps-vs-lv3-npc",
        action="store_true",
        help="Train one model per map against the first level-3 NPC profile on that map using the map-name player deck",
    )
    parser.add_argument("--eval-episodes", type=int, default=50, help="Episodes for post-train evaluation in batch modes")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_maps:
        for map_id in list_map_ids():
            print(map_id)
        return

    if args.list_decks:
        for idx, rowid in enumerate(list_deck_rowids()):
            print(f"{idx:03d}: {rowid}")
        return

    if args.list_map_decks:
        for map_id in list_map_ids():
            p1, p2 = auto_deck_rowids_for_map(map_id)
            extra = map_deck_candidates(map_id)[:6]
            print(f"{map_id}: P1={p1} P2={p2} candidates={','.join(extra)}")
        return

    if __package__:
        from .ppo_trainer import PPOConfig, PPOTrainer
        from .rl_env import TableturfRLEnv
    else:
        from ppo_trainer import PPOConfig, PPOTrainer
        from rl_env import TableturfRLEnv

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
    if args.train_all_maps_vs_lv3_npc:
        results = []
        for i, map_id in enumerate(list_map_ids()):
            npc_profiles = level_npc_profiles_for_map(level=2, map_id=map_id)
            if not npc_profiles:
                print(f"[skip] map={map_id} reason=no_lv3_npc_profile")
                continue
            npc = npc_profiles[0]
            p1_selector = player_map_deck_selector(map_id)
            bot_style = STYLE_MAP[npc["bot_style_raw"]]
            env = TableturfRLEnv(
                map_id=map_id,
                p1_deck_selector=p1_selector,
                p2_deck_selector=npc["p2_deck"],
                bot_style=bot_style,
                bot_level="high",
                seed=args.seed + i,
            )
            map_save_dir = str(Path(args.save_dir) / map_id)
            print(
                f"[train-all-maps-vs-lv3-npc] map={map_id} map_name={map_name_by_id(map_id)} "
                f"npc={npc['npc_rowid']} p1_deck={env.p1_deck_rowid} p2_deck={env.p2_deck_rowid} "
                f"bot={bot_style}/high save_dir={map_save_dir}"
            )
            trainer = PPOTrainer(
                env=env,
                config=cfg,
                device=args.device,
                save_dir=map_save_dir,
                run_name=f"train_{map_id}_vs_{npc['npc_rowid']}",
                run_config={
                    "mode": "train_all_maps_vs_lv3_npc",
                    "map_id": map_id,
                    "map_name": map_name_by_id(map_id),
                    "p1_deck": env.p1_deck_rowid,
                    "p2_deck": env.p2_deck_rowid,
                    "bot_style": bot_style,
                    "bot_level": "high",
                    "npc_rowid": npc["npc_rowid"],
                    "npc_name": npc["npc_name"],
                    "npc_level": 2,
                    "seed": args.seed + i,
                    "resume_checkpoint": args.resume_checkpoint or "",
                },
            )
            if args.resume_checkpoint:
                trainer.load_checkpoint(args.resume_checkpoint)
            trainer.train()
            win, draw, lose = run_eval(trainer.model, env, args.eval_episodes, torch.device(args.device))
            result = {
                "map_id": map_id,
                "map_name": map_name_by_id(map_id),
                "npc_rowid": npc["npc_rowid"],
                "npc_name": npc["npc_name"],
                "npc_level": 2,
                "p1_deck": env.p1_deck_rowid,
                "p2_deck": env.p2_deck_rowid,
                "bot_style": bot_style,
                "bot_level": "high",
                "eval_episodes": args.eval_episodes,
                "win_rate": win,
                "draw_rate": draw,
                "loss_rate": lose,
                "save_dir": map_save_dir,
            }
            results.append(result)
            Path(map_save_dir, "eval_summary.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"[eval] map={map_id} npc={npc['npc_rowid']} episodes={args.eval_episodes} "
                f"win_rate={win:.3f} draw_rate={draw:.3f} loss_rate={lose:.3f}"
            )
        batch_summary = {
            "mode": "train_all_maps_vs_lv3_npc",
            "total_maps": len(results),
            "eval_episodes": args.eval_episodes,
            "results": results,
        }
        batch_summary_path = Path(args.save_dir) / "batch_eval_summary.json"
        batch_summary_path.parent.mkdir(parents=True, exist_ok=True)
        batch_summary_path.write_text(json.dumps(batch_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[batch-summary] path={batch_summary_path}")
        return

    if args.train_all_maps:
        for i, map_id in enumerate(list_map_ids()):
            env = TableturfRLEnv(
                map_id=map_id,
                p1_deck_selector=args.p1_deck,
                p2_deck_selector=args.p2_deck,
                bot_style=args.bot_style,
                bot_level=args.bot_level,
                seed=args.seed + i,
            )
            map_save_dir = str(Path(args.save_dir) / map_id)
            print(
                f"[train-all-maps] map={map_id} "
                f"p1_deck={env.p1_deck_rowid} p2_deck={env.p2_deck_rowid} "
                f"save_dir={map_save_dir}"
            )
            trainer = PPOTrainer(
                env=env,
                config=cfg,
                device=args.device,
                save_dir=map_save_dir,
                run_name=f"train_{map_id}",
                run_config={
                    "map_id": map_id,
                    "p1_deck": env.p1_deck_rowid,
                    "p2_deck": env.p2_deck_rowid,
                    "bot_style": args.bot_style,
                    "bot_level": args.bot_level,
                    "seed": args.seed + i,
                    "resume_checkpoint": args.resume_checkpoint or "",
                },
            )
            if args.resume_checkpoint:
                trainer.load_checkpoint(args.resume_checkpoint)
            trainer.train()
        return

    env = TableturfRLEnv(
        map_id=args.map_id,
        p1_deck_selector=args.p1_deck,
        p2_deck_selector=args.p2_deck,
        bot_style=args.bot_style,
        bot_level=args.bot_level,
        seed=args.seed,
    )
    print(f"[train] map={args.map_id} p1_deck={env.p1_deck_rowid} p2_deck={env.p2_deck_rowid}")
    trainer = PPOTrainer(
        env=env,
        config=cfg,
        device=args.device,
        save_dir=args.save_dir,
        run_name=f"train_{args.map_id}",
        run_config={
            "map_id": args.map_id,
            "p1_deck": env.p1_deck_rowid,
            "p2_deck": env.p2_deck_rowid,
            "bot_style": args.bot_style,
            "bot_level": args.bot_level,
            "seed": args.seed,
            "resume_checkpoint": args.resume_checkpoint or "",
        },
    )
    if args.resume_checkpoint:
        trainer.load_checkpoint(args.resume_checkpoint)
    trainer.train()


if __name__ == "__main__":
    main()
