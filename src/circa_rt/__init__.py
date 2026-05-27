"""CIRCA-RT local experiment package."""

from .core import CIRCARTModel, CIRCARuntimeConfig, apply_circa_rt, apply_circa_rt_slack_admissible, fit_circa_rt, run_circa_rt
from .metrics import summarize_detection

__all__ = [
    "CIRCARTModel",
    "CIRCARuntimeConfig",
    "apply_circa_rt",
    "apply_circa_rt_slack_admissible",
    "fit_circa_rt",
    "run_circa_rt",
    "summarize_detection",
]
