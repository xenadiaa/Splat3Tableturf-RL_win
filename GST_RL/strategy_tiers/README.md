# Strategy Tiers

## low_granularity

Current stage:
- Use existing map + deck setup
- Self-play and NPC battle training

Place files in:
- `checkpoints/`
- `configs/`
- `logs/`
- `eval/`

## medium_granularity

Per-map and per-opponent stage:
- Deck search for strong matchup decks
- Build map-level near-optimal recommended decks
- Then train battle policy with searched/recommended decks

Place files in:
- `deck_search/`
- `recommended_decks/`
- `checkpoints/`
- `configs/`
- `logs/`
- `eval/`

## high_granularity

Hierarchical stage:
- High-level policy selects 15 cards
- Low-level policy plays the game
- Very large action space, high training cost

Place files in:
- `hierarchical_policy/`
- `checkpoints/`
- `configs/`
- `logs/`
- `eval/`

