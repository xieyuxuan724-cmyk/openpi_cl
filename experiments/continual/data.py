"""LIBERO data factory with an explicit normalization convention."""

import dataclasses

from openpi.training.config import LeRobotLiberoDataConfig


@dataclasses.dataclass(frozen=True)
class ContinualLiberoDataConfig(LeRobotLiberoDataConfig):
    normalization: str = "quantile"

    def create(self, assets_dirs, model_config):
        if self.normalization not in ("quantile", "zscore"):
            raise ValueError("Unknown normalization convention")
        return dataclasses.replace(
            super().create(assets_dirs, model_config),
            use_quantile_norm=self.normalization == "quantile",
        )
