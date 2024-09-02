"""The Coherent CAPTAIN Mills Detector."""

from typing import Dict, Callable
import torch
import os

from graphnet.models.detector.detector import Detector
from graphnet.constants import CCM_GEOMETRY_TABLE_DIR


class CCM(Detector):
    """CCM Detector class."""

    geometry_table_path = os.path.join(CCM_GEOMETRY_TABLE_DIR, "ccm.parquet")

    xyz = ["pmt_x", "pmt_y", "pmt_z"]
    rtheta = ["pmt_r", "pmt_theta"]
    pmt_index0_column = "pmt_index0"
    pmt_index1_column = "pmt_index1"
    pmt_type_column = "pmt_type"

    def feature_map(self) -> Dict[str, Callable]:
        """Normalize positional information."""
        feature_map = {
            "pmt_x": self._pmt_xyr,
            "pmt_y": self._pmt_xyr,
            "pmt_z": self._pmt_z,
            "pmt_r": self._pmt_xyr,
        }
        return feature_map

    def _pmt_xyr(self, x: torch.tensor) -> torch.tensor:
        return x / 96.0

    def _pmt_z(self, x: torch.tensor) -> torch.tensor:
        return x / 58.0
