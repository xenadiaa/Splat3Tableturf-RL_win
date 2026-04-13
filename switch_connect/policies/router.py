from __future__ import annotations

from typing import Optional

from switch_connect.policies.engine_policy import choose_action_engine
from switch_connect.policies.nn_policy import choose_action_nn_command, choose_action_nn_module
from vision_capture.state_types import ObservedState


def choose_action(
    obs: ObservedState,
    policy: str = "engine",
    style: Optional[str] = None,
    level: str = "high",
    nn_module: str = "",
    nn_command: str = "",
):
    if policy == "engine":
        return choose_action_engine(obs=obs, style=style, level=level)
    if policy == "nn-module":
        if not nn_module:
            raise ValueError("--nn-module is required when policy=nn-module")
        return choose_action_nn_module(obs=obs, module_ref=nn_module)
    if policy == "nn-command":
        if not nn_command:
            raise ValueError("--nn-command is required when policy=nn-command")
        return choose_action_nn_command(obs=obs, command=nn_command)
    raise ValueError("policy must be one of: engine, nn-module, nn-command")
