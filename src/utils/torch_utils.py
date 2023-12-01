import torch
from src.utils import pylogger
from typing import Tuple

log = pylogger.get_pylogger(__name__)


def load_checkpoint(
    model,
    checkpoint_path,
    allow_extra_keys=False,
    extra_key="state_dict",
    replace: Tuple = None,
    map_location="cpu",
):
    """Loads checkpoint into model with optional replacement of keys."""
    if not checkpoint_path:
        raise Exception("Checkpoint path needed!")

    log.info(f"Loading checkpoint! <checkpoint={checkpoint_path}>")

    state_dict = torch.load(checkpoint_path, map_location=map_location)
    if extra_key:
        state_dict = state_dict[extra_key]

    if replace:
        for key in list(state_dict.keys()):
            state_dict[key.replace(replace[0], replace[1])] = state_dict.pop(key)

    if allow_extra_keys:
        state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

    model.load_state_dict(state_dict, strict=True)

    return model
