"""I3Extractor class(es) for extracting specific, reconstructed features."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional
import numpy as np

from .i3extractor import I3Extractor
from graphnet.data.extractors.icecube.utilities.frames import (
    get_om_keys_and_pulseseries,
)
from graphnet.utilities.imports import has_icecube_package
from graphnet.data.extractors.icecube import I3FeatureExtractor

if has_icecube_package() or TYPE_CHECKING:
    from icecube import icetray  # pyright: reportMissingImports=false


class I3SummaryStatistics(I3FeatureExtractor):
    """Class for extracting summary statistics for IceCube-86."""

    def __init__(
        self,
        pulsemap: str,
        charge_quantiles: List[float],
        time_quantiles: List[float],
        exclusions: Optional[List[str]] = None,
    ):
        """Construct I3SummaryStatistics.

        Args:
            pulsemap: Name of the pulse (series) map for which to extract
                reconstructed features.
        """
        # Base class constructor
        super().__init__(pulsemap + "_summary_statistics")

        # Member variable(s)
        self._pulsemap = pulsemap
        self._charge_quantiles = charge_quantiles
        self._time_quantiles = time_quantiles
        self._exclusions = exclusions

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, List[Any]]:
        """Extract reconstructed features from `frame`.

        Args:
            frame: Physics (P) I3-frame from which to extract reconstructed
                features.

        Returns:
            Dictionary of reconstructed features for all pulses in `pulsemap`,
                in pure-python format.
        """
        padding_value: float = -1.0
        output: Dict[str, List[Any]] = {
            "charge": [],
            "time_first": [],
            "time_last": [],
            "charge_weighted_mean_time": [],
            "charge_weighted_std_time": [],
            "dom_x": [],
            "dom_y": [],
            "dom_z": [],
            "is_bright_dom": [],
            "is_bad_dom": [],
            "is_saturated_dom": [],
            "is_errata_dom": [],
            "event_time": [],
            "string": [],
            "pmt_number": [],
            "dom_number": [],
            "dom_type": [],
        }

        # Add quantiles
        for charge_quantile in self._charge_quantiles:
            output[f"charge_{charge_quantile}"] = []
        for time_quantile in self._time_quantiles:
            output[f"time_{time_quantile}"] = []

        # Get OM data
        if self._pulsemap in frame:
            om_keys, data = get_om_keys_and_pulseseries(
                frame,
                self._pulsemap,
                self._calibration,
            )
        else:
            self.warning_once(f"Pulsemap {self._pulsemap} not found in frame.")
            return output

        # Added these :
        bright_doms = None
        bad_doms = None
        saturation_windows = None
        calibration_errata = None
        if "BrightDOMs" in frame:
            bright_doms = frame.Get("BrightDOMs")

        if "BadDomsList" in frame:
            bad_doms = frame.Get("BadDomsList")

        if "SaturationWindows" in frame:
            saturation_windows = frame.Get("SaturationWindows")

        if "CalibrationErrata" in frame:
            calibration_errata = frame.Get("CalibrationErrata")

        event_time = frame["I3EventHeader"].start_time.mod_julian_day_double

        # Get global info for offsets
        charges = []
        times = []
        dom_charges = {}
        dom_times = {}
        for om_key in om_keys:
            # Loop over pulses for each OM
            pulses = data[om_key]

            dom_charges[om_key] = []
            dom_times[om_key] = []
            for pulse in pulses:
                dom_charges[om_key].append(pulse.charge)
                dom_times[om_key].append(pulse.time)
            charges.extend(dom_charges[om_key])
            times.extend(dom_times[om_key])
            dom_charges[om_key] = np.asarray(dom_charges[om_key])
            dom_times[om_key] = np.asarray(dom_times[om_key])

        charges = np.asarray(charges)
        times = np.asarray(times)

        global_charge_wgt_mean_time = np.average(times, weights=charges)

        for om_key in om_keys:
            # Common values for each OM
            x = self._gcd_dict[om_key].position.x
            y = self._gcd_dict[om_key].position.y
            z = self._gcd_dict[om_key].position.z
            area = self._gcd_dict[om_key].area
            # rde = self._get_relative_dom_efficiency(
            #     frame, om_key, padding_value
            # )

            string = om_key[0]
            dom_number = om_key[1]
            pmt_number = om_key[2]
            dom_type = self._gcd_dict[om_key].omtype

            # DOM flags
            if bright_doms:
                is_bright_dom = 1 if om_key in bright_doms else 0
            else:
                is_bright_dom = int(padding_value)

            if bad_doms:
                is_bad_dom = 1 if om_key in bad_doms else 0
            else:
                is_bad_dom = int(padding_value)

            if saturation_windows:
                is_saturated_dom = 1 if om_key in saturation_windows else 0
            else:
                is_saturated_dom = int(padding_value)

            if calibration_errata:
                is_errata_dom = 1 if om_key in calibration_errata else 0
            else:
                is_errata_dom = int(padding_value)

            total_dom_charge = sum(dom_charges[om_key])
            rel_times = dom_times[om_key] - global_charge_wgt_mean_time

            charge_weighted_mean_rel_time = np.average(
                rel_times, weights=dom_charges[om_key]
            )
            charge_weighted_std_rel_time = weighted_std(
                rel_times, weights=dom_charges[om_key]
            )

            output["charge"].append(total_dom_charge)
            output["time_first"].append(rel_times[0])
            output["time_last"].append(rel_times[-1])

            output["charge_weighted_mean_time"].append(
                charge_weighted_mean_rel_time
            )
            output["charge_weighted_std_time"].append(
                charge_weighted_std_rel_time
            )

            for time_quantile in self._time_quantiles:
                output[f"time_{time_quantile}"].append(
                    weighted_quantile(
                        rel_times,
                        weights=dom_charges[om_key],
                        quantile=time_quantile,
                    )
                )

            output["dom_x"].append(x)
            output["dom_y"].append(y)
            output["dom_z"].append(z)
            # ID's
            output["string"].append(string)
            output["pmt_number"].append(pmt_number)
            output["dom_number"].append(dom_number)
            output["dom_type"].append(dom_type)
            # DOM flags
            output["is_bright_dom"].append(is_bright_dom)
            output["is_bad_dom"].append(is_bad_dom)
            output["is_saturated_dom"].append(is_saturated_dom)
            output["is_errata_dom"].append(is_errata_dom)
            output["event_time"].append(event_time)

        return output


def weighted_quantile(x, weights, quantile=0.68):
    """Compute weighted quantile.

    Parameters
    ----------
    x : list or numpy.ndarray
        The data for which to compute the quantile
    weights : list or numpy.ndarray
        The weights for x.
    quantile : float, optional
        The quantile to compute.

    Returns
    -------
    float
        The weighted quantile
    """
    if weights is None:
        weights = np.ones_like(x)

    x = np.asarray(x)
    weights = np.asarray(weights)

    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    weights_sorted = weights[sorted_indices]
    cum_weights = np.cumsum(weights_sorted) / np.sum(weights)
    mask = cum_weights >= quantile

    return x_sorted[mask][0]


def weighted_std(x, weights=None):
    """" Weighted std deviation.

    Source
    ------
    http://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf

    returns 0 if len(x)==1

    Parameters
    ----------
    x : list or numpy.ndarray
        Description
    weights : None, optional
        Description

    Returns
    -------
    float
        Weighted standard deviation
    """
    if len(x) == 1:
        return 0

    if weights is None:
        return np.std(x, ddof=1)

    x = np.asarray(x)
    weights = np.asarray(weights)

    w_mean_x = np.average(x, weights=weights)
    n = len(weights[weights != 0])

    s = (
        n
        * np.sum(weights * (x - w_mean_x) * (x - w_mean_x))
        / ((n - 1) * np.sum(weights))
    )
    return np.sqrt(s)
