# switch_connect

Realtime Switch bridge for:
- policy selection
- action mapping
- virtual gamepad / serial output

Pipeline:
`ObservedState -> policy(engine/nn) -> action -> virtual_gamepad -> serial`

Repository split by responsibility:
- `vision_capture`: capture-card device access and screenshots
- `tableturf_vision`: Tableturf-specific image parsing and judge tools
- `switch_connect`: strategy and controller output only

## Layout

- `virtual_gamepad/`
  - `input_mapper.py`: action -> button step mapping
  - `serial_controller.py`: 0xFF remote mode sender
  - `smart_program_compat.py`: AutoController smart-program command encoding (0xFE)
- `strategy_mapper.py`
  - uses existing engine rules/scoring:
    - `src.engine.env_core.legal_actions`
    - `src.engine.env_core._score_bot_action`
- `policies/`
  - `router.py`: policy switch
  - `engine_policy.py`: built-in engine strategy
  - `nn_policy.py`: neural policy adapters (python module or external command)
- `bridge_runner.py`
  - CLI entrypoint for json state, action selection, optional serial output

## Quick Start

```bash
cd ..
python3 switch_connect/bridge_runner.py \
  --state-json switch_connect/examples/state_example.json \
  --print-steps
```

Capture and vision tools have been moved out of this package.
Use:
- `vision_capture`
- `tableturf_vision`

Use NN policy (python module callable):

```bash
python3 switch_connect/bridge_runner.py \
  --state-json switch_connect/examples/state_example.json \
  --policy nn-module \
  --nn-module switch_connect.policies.examples.simple_nn_policy:infer \
  --print-steps
```

Use NN policy (external process):

```bash
python3 switch_connect/bridge_runner.py \
  --state-json switch_connect/examples/state_example.json \
  --policy nn-command \
  --nn-command "python3 your_infer.py" \
  --print-steps
```

Send to serial:

```bash
python3 switch_connect/bridge_runner.py \
  --state-json switch_connect/examples/state_example.json \
  --serial-port /dev/cu.SLAB_USBtoUART \
  --print-steps
```

Send AutoController-compatible smart sequence (0xFE mode):

```bash
python3 switch_connect/virtual_gamepad/send_smart_sequence.py \
  --serial-port /dev/cu.SLAB_USBtoUART \
  --commands "A,1,Nothing,20,DRight,1,ASpam,10"
```

Manual selection with arrow keys (serial port):

```bash
python3 switch_connect/virtual_gamepad/send_smart_sequence.py \
  --pick-serial \
  --commands "A,1,Nothing,20,DRight,1,ASpam,10"
```

## Policy I/O contract

- policy input: `ObservedState` JSON
- policy output: Action JSON object:
  - `player` (`"P1"`)
  - `card_number` (int or null)
  - `pass_turn` (bool)
  - `use_sp_attack` (bool)
  - `rotation` (0..3)
  - `x`, `y` (int, nullable when pass)

## Smart Program Compatibility

- Firmware mode:
  - `0xFF + uint32`: remote bit-control
  - `0xFE + 30*(char + uint16 duration)`: smart sequence table
- Implemented from `AutoController_swsh/SourceCode/Bots/Others_SmartProgram/Others_SmartProgram.c`
- Compatible tokens include:
  - `A/B/X/Y/L/R/ZL/ZR/Home/Capture/Plus/Minus`
  - `DUp/DDown/DLeft/DRight`
  - `LUp/LDown/LLeft/LRight` and diagonals
  - `ASpam/BSpam/Loop`
  - combo tokens such as `DRightR`, `ZLBX`, `ZLA`, `BY`, `LUpClick`
