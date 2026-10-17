"""Mamba-Jev: backbone Mamba + teste lineari in forma chiusa, con calibrazione LOO e rejection dalle stesse statistiche."""
from .config import MambaJevConfig
from .loss import CombinedJevLoss
from .model import MambaJevModel
from .pipeline import JevFit, fit_jev_heads
from .solver import ExactHeadSolver, HeadFit
from .train import TrainConfig, evaluate, train
from .types import JevOutput, JevTargets
from .uncertainty import Calibration, DomainModel, fit_platt, fit_temperature

__version__ = "0.2.0"

__all__ = [
    "MambaJevConfig",
    "MambaJevModel",
    "CombinedJevLoss",
    "ExactHeadSolver",
    "HeadFit",
    "TrainConfig",
    "train",
    "evaluate",
    "JevOutput",
    "JevTargets",
    "JevFit",
    "fit_jev_heads",
    "Calibration",
    "DomainModel",
    "fit_temperature",
    "fit_platt",
]
