from __future__ import annotations

import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from spectral_model import get_spectrum_from_neural_net
from convolve import *
from scipy.interpolate import interp1d
import os
from dataclasses import dataclass, field
from scipy.optimize import minimize as scipy_minimize
from lmfit import Parameters, minimize, fit_report

# Created by storm at 03.03.25

@dataclass(slots=True)
class PayneParams:
    labels: list[str]
    payne_coeffs: tuple
    wavelength_payne: np.ndarray
    x_min: list
    x_max: list
    resolution_val: float | None = None

@dataclass(slots=True)
class Parameter:
    label_name: str
    value: float
    std: float = -99            # default: not set
    fit: bool = True            # default: include in optimisation
    min_value: float | None = None
    max_value: float | None = None

    def bounds(self) -> tuple[float, float]:
        return self.min_value, self.max_value

    def fmt(self) -> str:
        flag = "fit " if self.fit else "fix "
        return f"{flag} {self.value:10.3f} ± {self.std:7.3f}"

@dataclass(slots=True)
class LSQSetup:
    p0: list[float]
    input_values: list[float | None]
    bounds: tuple[list[float], list[float]]
    labels: list[str]               # labels actually being fitted, same order as p0

@dataclass(slots=True)
class StellarParameters:
    teff: Parameter
    logg: Parameter
    feh:  Parameter
    vmic: Parameter
    vsini: Parameter
    vmac: Parameter
    doppler_shift: Parameter
    abundances: dict[str, Parameter] = field(default_factory=dict)

    _core_map = {
        "teff": "teff",  # label → attribute name
        "logg": "logg",
        "feh": "feh",
        "vmic": "vmic",
        "vsini": "vsini",
        "vmac": "vmac",
        "doppler_shift": "doppler_shift",
    }

    def __str__(self) -> str:
        # Column widths
        name_w, val_w = 15, 20
        header = f"{'Label':{name_w}} | {'Mode & Value':{val_w}}\n" + "-" * (name_w + val_w + 3)

        rows: list[str] = []

        # core labels in fixed order
        for lab in ("teff", "logg", "feh", "vmic", "vsini", "vmac", "doppler_shift"):
            param: Parameter = getattr(self, lab)
            rows.append(f"{lab:{name_w}} | {param.fmt()}")

        # abundances sorted alphabetically
        for lab, param in sorted(self.abundances.items()):
            rows.append(f"{lab:{name_w}} | {param.fmt()}")

        return "\n".join([header, *rows])

    def build_lsq_inputs(
        self,
        payne_labels,
        labels_to_fit
    ) -> LSQSetup:
        """
        Assemble the vectors required by `scipy.optimize.curve_fit`.

        Parameters
        ----------
        payne_labels
            Full label list used by The Payne *excluding* vsini/vmac/doppler_shift.
        labels_to_fit
            Iterable of label names that should be optimised.

        Returns
        -------
        LSQSetup
            p0, input vector (with None placeholders), per-parameter bounds,
            and the list of labels whose order matches p0.
        """
        # (1) final label sequence ------------------------------------------------
        labels = list(payne_labels) + ["vsini", "vmac", "doppler_shift"]

        # (2) fast lookup helpers -------------------------------------------------
        fit_set: set[str] = set(labels_to_fit)
        p0: list[float] = []
        pmin: list[float] = []
        pmax: list[float] = []
        input_values: list[float | None] = []
        fitted_labels: list[str] = []

        # (3) build vectors -------------------------------------------------------
        for lab in labels:
            # select the Parameter object
            if lab in self._core_map:
                param: Parameter = getattr(self, self._core_map[lab])
            else:
                # safe even if abundances is regular dict: key order comes from labels[]
                param = self.abundances.get(lab)
                if param is None:
                    raise KeyError(f"StellarParameters missing abundance '{lab}'")

            # decide whether we fit this label
            is_fitted = param.fit and lab in fit_set
            input_values.append(None if is_fitted else param.value)

            if is_fitted:
                p0.append(param.value)
                lo, hi = param.bounds()
                pmin.append(lo if lo is not None else -np.inf)
                pmax.append(hi if hi is not None else  np.inf)
                fitted_labels.append(lab)

        return LSQSetup(
            p0=p0,
            input_values=input_values,
            bounds=(pmin, pmax),
            labels=fitted_labels,
        )

    def iter_params(self):
        """Yield (label, Parameter) for core labels *then* abundances and finally vsini/vmac/doppler_shift."""
        for lab in ("teff", "logg", "feh", "vmic"):
            yield lab, getattr(self, lab)
        yield from self.abundances.items()
        for lab in ("vsini", "vmac", "doppler_shift"):
            yield lab, getattr(self, lab)

    # ––––– public helper –––––
    def to_value_sigma_dicts(self, rename_map: dict[str, str] | None = None, only_fitted: bool = True):
        """
        Return two dicts: {label: value}, {label: std}.
        *rename_map* lets you rename keys on the fly.
        """
        ren = (lambda k: rename_map.get(k, k)) if rename_map else (lambda k: k)
        vals, sigs = {}, {}
        for lab, p in self.iter_params():
            if not only_fitted or p.fit:
                key = ren(lab)
                vals[key] = p.value
                sigs[key] = p.std
        return vals, sigs

def calculate_equivalent_width(wavelength: np.ndarray, normalised_flux: np.ndarray, left_bound: float, right_bound: float, max_gap_A=0.5, assume_sorted=True) -> float:
    """
    Calculates the equivalent width of a line based on the input parameters
    :param wavelength: Wavelength array
    :param normalised_flux: Normalised flux array
    :param left_bound: Left bound of the line
    :param right_bound: Right bound of the line
    :param max_gap_A: Maximum gap allowed between the bound and the nearest data point to interpolate the continuum
    :return: Equivalent width of the line in mA
    """

    w = wavelength
    f = normalised_flux

    if not assume_sorted:
        # keep points that bracket the region
        order = np.argsort(w)
        w = w[order]
        f = f[order]

    if right_bound <= left_bound or right_bound <= w[0] or left_bound >= w[-1]:
        return -9999

    # mask interior points
    m = (w > left_bound) & (w < right_bound)
    wi = w[m]
    fi = f[m]

    # Default: continuum at boundaries
    fl = 1.0
    fr = 1.0

    # Left boundary: only interpolate if there is a point sufficiently close on the right
    if wi.size > 0:
        nearest_right = wi[0]
        if (nearest_right - left_bound) <= max_gap_A:
            fl = np.interp(left_bound, w, f, left=1.0, right=1.0)

        nearest_left = wi[-1]
        if (right_bound - nearest_left) <= max_gap_A:
            fr = np.interp(right_bound, w, f, left=1.0, right=1.0)
    else:
        # No interior points: only meaningful if bounds are within data and close; otherwise return sentinel
        return -9999

    w2 = np.concatenate(([left_bound], wi, [right_bound]))
    f2 = np.concatenate(([fl], fi, [fr]))

    # based on the numpy version use trapezoid or trapz
    if hasattr(np, "trapezoid"):
        ew = np.trapezoid(1.0 - f2, w2)
    else:
        ew = np.trapz(1.0 - f2, w2)
    return ew * 1000.0  # convert to mA

def process_spectra(wavelength_payne, wavelength_obs, flux_obs, err_obs, h_line_cores, h_line_core_mask_dlam=0.5, extra_payne_cut=10):
    """
    Loads the observed spectrum, cuts out unnecessary parts, and processes it to be used with the Payne model.
    :param wavelength_payne: Wavelength array corresponding to the Payne model. Rest frame.
    :param wavelength_obs: Wavelengths of the observed spectrum. Not in the rest frame.
    :param flux_obs: Fluxes of the observed spectrum.
    :param h_line_cores: List of hydrogen line core wavelengths to mask out.
    :param h_line_core_mask_dlam: The width of the mask around the hydrogen line cores in Angstroms.
    :param extra_payne_cut: Extra cut around the Payne model wavelength range in Angstroms.
    :return: Returns the processed wavelength and flux arrays.
    """
    # remove any negative fluxes and fluxes > 1.2
    mask = (flux_obs > 0.0) & (flux_obs < 1.2)
    wavelength_obs = wavelength_obs[mask]
    flux_obs = flux_obs[mask]
    if err_obs is not None:
        err_obs = err_obs[mask]

    # mask out the hydrogen line cores; to be fair not fully correct because h_line_cores are in rest frame, but we
    # assume that the observed spectrum is close enough to the rest frame
    mask = np.all(np.abs(wavelength_obs[:, None] - h_line_cores) > h_line_core_mask_dlam, axis=1)
    wavelength_obs = wavelength_obs[mask]
    flux_obs = flux_obs[mask]
    if err_obs is not None:
        err_obs = err_obs[mask]

    # cut the observed spectrum to the Payne model wavelength range; no need to carry around the whole spectrum
    l_cut = (wavelength_obs > wavelength_payne[0] - extra_payne_cut) & (wavelength_obs < wavelength_payne[-1] + extra_payne_cut)
    wavelength_obs = wavelength_obs[l_cut]
    flux_obs = flux_obs[l_cut]
    if err_obs is not None:
        err_obs = err_obs[l_cut]

    if err_obs is None:
        err_obs = np.ones_like(wavelength_obs) / 100
    return wavelength_obs, flux_obs, err_obs

def calculate_vturb(teff: float, logg: float, met: float) -> float:
    """
    Calculates micro turbulence based on the input parameters
    :param teff: Temperature in kelvin
    :param logg: log(g) in dex units
    :param met: metallicity [Fe/H] scaled by solar
    :return: micro turbulence in km/s
    """
    t0 = 5500.
    g0 = 4.
    tlim = 5000.
    glim = 3.5

    delta_logg = logg - g0

    if logg >= glim:
        # dwarfs
        if teff >= tlim:
            # hot dwarfs
            delta_t = teff - t0
        else:
            # cool dwarfs
            delta_t = tlim - t0

        v_mturb = (1.05 + 2.51e-4 * delta_t + 1.5e-7 * delta_t**2 - 0.14 * delta_logg - 0.005 * delta_logg**2 +
                   0.05 * met + 0.01 * met**2)

    else:
        # giants
        delta_t = teff - t0

        v_mturb = (1.25 + 4.01e-4 * delta_t + 3.1e-7 * delta_t**2 - 0.14 * delta_logg - 0.005 * delta_logg**2 +
                   0.05 * met + 0.01 * met**2)

    return v_mturb

def apply_on_segments(
        wavelength,            # 1-D array-like
        spectrum,              # 1-D array-like, same length
        func,                  # callable(wl_seg, sp_seg, *a, **kw) → (wl_out, sp_out)
        *func_args,            # forwarded positional args (e.g. vsini)
        spacing_tolerance=2.0, # gap = diff > tol × median_spacing
        assume_sorted=True,    # set False if the array is unsorted
        **func_kwargs):        # forwarded keyword args
    """
    Call `func` independently on each uniformly-spaced wavelength segment.
    Basically, it applies convolution on each spectra segment that is separated by a "large" gap.
    It is faster than applying `func` on the whole array at once, especially for large arrays.

    Returns
    -------
    wavelength_out, spectrum_out : np.ndarray
        Concatenated outputs in the original order.
    """
    wl = np.asarray(wavelength)
    sp = np.asarray(spectrum)

    if wl.ndim != 1 or wl.shape != sp.shape:
        raise ValueError("`wavelength` and `spectrum` must be 1-D and equally long")

    # sort once if needed (keeps wavelengths increasing)
    if not assume_sorted:
        order = np.argsort(wl)
        wl, sp = wl[order], sp[order]

    # identify “large” gaps
    diffs           = np.diff(wl)
    median_spacing  = np.median(diffs)
    gap_locations   = np.where(diffs > spacing_tolerance * median_spacing)[0]

    # segment boundaries (inclusive start, exclusive end)
    bounds = np.concatenate(([0], gap_locations + 1, [len(wl)]))

    w_out, s_out = [], []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        wl_seg, sp_seg = wl[start:end], sp[start:end]
        wl_conv, sp_conv = func(wl_seg, sp_seg, *func_args, **func_kwargs)
        w_out.append(np.asarray(wl_conv))
        s_out.append(np.asarray(sp_conv))

    return np.concatenate(w_out), np.concatenate(s_out)


def refit_continuum_spectrum(
        wavelength_obs,        # 1-D array-like
        flux_obs,              # 1-D array-like, same length
        err_obs_variance,    # 1-D array-like, same length
        flux_synthetic,        # 1-D array-like, same length, on the same grid
        spacing_tolerance=2.0, # gap = diff > tol × median_spacing
        assume_sorted=True,    # set False if the array is unsorted
        **func_kwargs):        # forwarded keyword args
    """

    Returns
    -------
    spectrum_out : np.ndarray
        Concatenated outputs in the original order.
    """
    wl = np.asarray(wavelength_obs)
    sp = np.asarray(flux_obs)

    if wl.ndim != 1 or wl.shape != sp.shape:
        raise ValueError("`wavelength` and `spectrum` must be 1-D and equally long")

    # sort once if needed (keeps wavelengths increasing)
    if not assume_sorted:
        order = np.argsort(wl)
        wl, sp = wl[order], sp[order]

    # identify “large” gaps
    diffs           = np.diff(wl)
    median_spacing  = np.median(diffs)
    gap_locations   = np.where(diffs > spacing_tolerance * median_spacing)[0]

    # segment boundaries (inclusive start, exclusive end)
    bounds = np.concatenate(([0], gap_locations + 1, [len(wl)]))

    s_out = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        wl_seg, sp_seg, err_seg, synthetic_seg = wl[start:end], sp[start:end], err_obs_variance[start:end], flux_synthetic[start:end]
        sp_conv = refit_continuum_spectrum_one_segment(wl_seg, sp_seg, err_seg, synthetic_seg)
        s_out.append(np.asarray(sp_conv))

    return np.concatenate(s_out)

def refit_continuum_spectrum_hr_windows(
        wavelength_obs,        # 1-D array-like
        flux_obs,              # 1-D array-like, same length
        err_obs_variance,    # 1-D array-like, same length
        flux_synthetic,        # 1-D array-like, same length, on the same grid
        assume_sorted=True,    # set False if the array is unsorted
        **func_kwargs):        # forwarded keyword args
    """

    Returns
    -------
    spectrum_out : np.ndarray
        Concatenated outputs in the original order.
    """
    wl = np.asarray(wavelength_obs)
    sp = np.asarray(flux_obs)

    if wl.ndim != 1 or wl.shape != sp.shape:
        raise ValueError("`wavelength` and `spectrum` must be 1-D and equally long")

    # sort once if needed (keeps wavelengths increasing)
    if not assume_sorted:
        order = np.argsort(wl)
        wl, sp = wl[order], sp[order]

    # gaps at HR windows:
    gap_locations = np.array([4500, 5800])

    # segment boundaries (inclusive start, exclusive end)
    bounds = np.concatenate(([0], gap_locations + 1, [len(wl)]))

    s_out = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        wl_seg, sp_seg, err_seg, synthetic_seg = wl[start:end], sp[start:end], err_obs_variance[start:end], flux_synthetic[start:end]
        if len(wl_seg) == 0:
            continue
        sp_conv = refit_continuum_spectrum_one_segment(wl_seg, sp_seg, err_seg, synthetic_seg)
        s_out.append(np.asarray(sp_conv))

    return np.concatenate(s_out)


def apply_continuum_correction(wavelength, wavelength_0, flux, continuum_slope_coef, continuum_intercept_coef):
    """
    Applies correction to the continuum in the form flux_out = flux + (1 - ((wavelength - wavelength_0) * slope + intercept))
    :param wavelength: wavelength array
    :param wavelength_0: from where to start normalisation (0-point)
    :param flux: normalised flux to normalise
    :param continuum_slope_coef: slope of the normalisation
    :param continuum_intercept_coef: intercept
    :return:
    """
    return flux * (1 - (continuum_slope_coef * (wavelength - wavelength_0) + continuum_intercept_coef))

def refit_continuum_spectrum_one_segment(wavelength_obs, flux_obs, error_obs_variance, flux_synthetic, continuum_start=None):
    if continuum_start is None:
        continuum_start = wavelength_obs[0]
    #function_args = (wavelength_obs, flux_obs, error_obs_variance, flux_synthetic)
    #minimize_options = {'maxiter': 100, 'disp': False}
    #res = scipy_minimize(calc_chi_sq_continuum, x0=np.asarray([0.001, 1]), args=function_args, bounds=[(-0.1, 0.1), (0.8, 1.2)],
    #               method='L-BFGS-B', options=minimize_options)
#
    #continuum_coeff_a = res.x[0]
    #continuum_coeff_b = res.x[1]
    
    poly_fit = np.polyfit(
        wavelength_obs - continuum_start, 1. - flux_obs / flux_synthetic, 1, w=1/np.sqrt(error_obs_variance)   
    )
    continuum_coeff_a = poly_fit[0]
    continuum_coeff_b = poly_fit[1]
    

    #print(f"Refit continuum: slope={continuum_coeff_a:.6f}, intercept={continuum_coeff_b:.6f}")

    flux_norm_fitted = apply_continuum_correction(wavelength_obs, continuum_start, np.copy(flux_synthetic),
                                                  continuum_coeff_a, continuum_coeff_b)

    return flux_norm_fitted


def calc_chi_sq_continuum(param, wavelength_fitted, flux_obs_fitted, error_obs_variance, flux_norm_fitted):
    continuum_coeff_a = param[0]
    continuum_coeff_b = param[1]
    flux_norm_fitted = apply_continuum_correction(wavelength_fitted, wavelength_fitted[0], np.copy(flux_norm_fitted),
                                                  continuum_coeff_a, continuum_coeff_b)

    return np.sum(np.square(flux_obs_fitted - flux_norm_fitted) / error_obs_variance)


def load_payne(path_model):
    """
    Loads the Payne model coefficients from a .npz file.
    :param path_model: Path to the .npz file containing the Payne model coefficients.
    :return: Returns a tuple containing the coefficients and the wavelength array.
    """
    tmp = np.load(path_model)
    w_array_0 = tmp["w_array_0"]
    w_array_1 = tmp["w_array_1"]
    w_array_2 = tmp["w_array_2"]
    w_array_3 = tmp["w_array_3"]
    b_array_0 = tmp["b_array_0"]
    b_array_1 = tmp["b_array_1"]
    b_array_2 = tmp["b_array_2"]
    b_array_3 = tmp["b_array_3"]
    x_min = tmp["x_min"]
    x_max = tmp["x_max"]
    # wavelength is the wavelength array corresponding to the Payne model in AA
    wavelength = tmp["wavelength"]
    # labels are the label names corresponding to the Payne model, e.g. "teff", "logg", "feh", etc.
    labels = list(tmp["label_names"])
    tmp.close()
    # w_array are the weights, b_array are the biases
    # x_min and x_max are the minimum and maximum values for scaling the labels
    payne_coeffs = (w_array_0, w_array_1, w_array_2, w_array_3,
                    b_array_0, b_array_1, b_array_2, b_array_3,
                    x_min, x_max)
    return payne_coeffs, wavelength, labels


def make_model_spectrum(payne_coeffs, wavelength_payne, resolution_val=None,
            pixel_limits=None, flux_obs=None, refit_continuum=False):
    """
    Creates a model spectrum function for curve fitting.
    :param payne_coeffs: Coefficients from the Payne model. Typically first few arrays are weights and biases,
    while the last two are x_min and x_max for scaling.
    :param wavelength_payne: Wavelength array corresponding to the Payne model. Rest frame.
    Example: [10, None, 5] would mean that the first parameter is fixed at 10, the second is free to fit, and the third is fixed at 5.
    :param resolution_val: Resolution value for convolution of Payne spectrum, if applicable.
    :param pixel_limits: A boolean mask for the pixel limits, if applicable. It masks the payne array to only
    include pixels that are within the limits.
    It can take three different types:
    1. Full of lists/tuples with wavelength limits, e.g. [(5000, 5005), (6000, 6005)] would mean that only pixels
    between 5000 and 5005 and between 6000 and 6005 are used.
    2. length is 2, with limits where to use: E.g. [5000, 5005] would mean that only pixels between 5000 and 5005 are used.
    3. Same length as the wavelength_payne, mask of the pixels to mask: E.g. [True, False, True] would mean that the
    first and third pixels are included, while the second is excluded.
    :param flux_obs: Observations, but only for plotting purposes, not for fitting.
    :return: Callable function that takes wavelength_obs and parameters to fit, and returns the model spectrum.
    """
    def model_spectrum(wavelength_obs, spectra_params):
        spectra_params = np.asarray(spectra_params, dtype=float).copy()

        vsini = spectra_params[-3]
        vmac = spectra_params[-2]
        doppler_shift = spectra_params[-1]

        real_labels = spectra_params[:-3].copy()

        # if vmic is 99, then scale with teff/logg/feh
        # if we pass vmic = 99 (i.e. not None), it is not fitted. But sometimes we want to calculate it based on the
        # empirical formula for the speed. Citation: Bergemann & Hoppe (LRCA, 2025)
        if real_labels[3] >= 98:
            real_labels[3] = calculate_vturb(real_labels[0] * 1000, real_labels[1], real_labels[2])

        # scale the labels to the Payne coefficients and get the spectrum
        scaled_labels = (real_labels - payne_coeffs[-2]) / (payne_coeffs[-1] - payne_coeffs[-2]) - 0.5
        spec_payne = get_spectrum_from_neural_net(
            scaled_labels=scaled_labels,
            NN_coeffs=payne_coeffs,
            pixel_limits=pixel_limits
        )

        wavelength_payne_ = wavelength_payne.astype(np.float64)

        if vmac > 0:
            wavelength_payne_, spec_payne = apply_on_segments(
                wavelength_payne_,
                spec_payne,
                conv_macroturbulence,
                vmac
            )
        if vsini > 0:
            wavelength_payne_, spec_payne = apply_on_segments(
                wavelength_payne_,
                spec_payne,
                conv_rotation,
                vsini
            )
        if resolution_val is not None:
            wavelength_payne_, spec_payne = conv_res(wavelength_payne_, spec_payne, resolution_val)

        # apply doppler shift
        if doppler_shift != 0:
            wavelength_payne_ = wavelength_payne_ * (1 + (doppler_shift / 299792.))

        f_interp = interp1d(
            wavelength_payne_,
            spec_payne,
            kind='linear',
            bounds_error=False,
            fill_value=1.0
        )
        interpolated_spectrum = f_interp(wavelength_obs)

        # debug: preview the interpolated spectrum
        # calculate chi-squared
        #if flux_obs is not None and False:
        #    chi_squared = np.sum((interpolated_spectrum - flux_obs) ** 2)
        #    print(params_to_fit[0], chi_squared)
        #    plt.figure(figsize=(14, 7))
        #    plt.title(params_to_fit[0])
        #    plt.scatter(wavelength_obs, flux_obs, s=3, color='k')
        #    plt.scatter(wavelength_obs, interpolated_spectrum, s=2, color='r')
        #    plt.show()

        if flux_obs is not None and refit_continuum:
            interpolated_spectrum = refit_continuum_spectrum_hr_windows(
                wavelength_obs,
                flux_obs,
                np.ones(np.size(wavelength_obs)) * 0.01,
                interpolated_spectrum
            )

        return interpolated_spectrum

    return model_spectrum

def scale_back(x, x_min, x_max, label_name=None):
    """
    Scales back the input values from the Payne model to the original scale.
    :param x: Values to scale back, typically coefficients from the Payne model.
    :param x_min: Minimum values for scaling, corresponding to the Payne model.
    :param x_max: Maximum values for scaling, corresponding to the Payne model.
    :param label_name: Optional label name to adjust the scaling for specific labels, e.g. "teff".
    :return: Returns the scaled back values as a list.
    """
    x = np.array(x)
    x_min = np.array(x_min)
    x_max = np.array(x_max)
    return_value = (x + 0.5) * (x_max - x_min) + x_min
    # if teff, the teff is typically in kK, so we scale it back to K
    if label_name == "teff":
        return_value = return_value * 1000
    return list(return_value)


def cut_to_just_lines(wavelength_obs, flux_obs, err_obs, wavelength_payne, lines_to_use, stellar_rv, obs_cut_aa=0.5, payne_cut_aa=0.75):
    # cut so that we only take the lines we want
    masks = []
    masks_payne = []

    if type(obs_cut_aa) is not list:
        obs_cut_aa = [obs_cut_aa] * len(lines_to_use)
    if type(payne_cut_aa) is not list:
        payne_cut_aa = [payne_cut_aa] * len(lines_to_use)

    if len(obs_cut_aa) == 1:
        obs_cut_aa = obs_cut_aa * len(lines_to_use)
    if len(payne_cut_aa) == 1:
        payne_cut_aa = payne_cut_aa * len(lines_to_use)

    wavelength_obs_rv_corrected = wavelength_obs / (1 + (stellar_rv / 299792.))

    for line, obs_cut_aa_one, payne_cut_aa_one in zip(lines_to_use, obs_cut_aa, payne_cut_aa):
        mask_one = (wavelength_obs_rv_corrected > line - obs_cut_aa_one) & (wavelength_obs_rv_corrected < line + obs_cut_aa_one)
        masks.append(mask_one)

        mask_payne = (wavelength_payne > line - payne_cut_aa_one) & (wavelength_payne < line + payne_cut_aa_one)
        masks_payne.append(mask_payne)
    # apply masks
    combined_mask = np.array(masks).any(axis=0)
    combined_mask_payne_ = np.array(masks_payne).any(axis=0)
    wavelength_obs_cut_to_lines_ = wavelength_obs[combined_mask]
    flux_obs_cut_to_lines_ = flux_obs[combined_mask]

    if err_obs is not None:
        err_obs_cut_to_lines_ = err_obs[combined_mask]
    else:
        err_obs_cut_to_lines_ = err_obs

    if combined_mask_payne_ is not None:
        wavelength_payne_cut_ = wavelength_payne[combined_mask_payne_]
    else:
        wavelength_payne_cut_ = wavelength_payne

    return wavelength_obs_cut_to_lines_, flux_obs_cut_to_lines_, err_obs_cut_to_lines_, wavelength_payne_cut_, combined_mask_payne_

def create_default_stellar_parameters(payne_parameters: PayneParams):
    p0 = scale_back([0] * (len(payne_parameters.labels)), payne_parameters.x_min, payne_parameters.x_max)

    stellar_parameters = StellarParameters(
        teff=Parameter(value=p0[0], std=-99, fit=True, min_value=payne_parameters.x_min[0], max_value=payne_parameters.x_max[0], label_name="teff"),
        logg=Parameter(value=p0[1], std=-99, fit=True, min_value=payne_parameters.x_min[1], max_value=payne_parameters.x_max[1], label_name="logg"),
        feh=Parameter(value=p0[2], std=-99, fit=True, min_value=payne_parameters.x_min[2], max_value=payne_parameters.x_max[2], label_name="feh"),
        vmic=Parameter(value=p0[3], std=-99, fit=True, min_value=payne_parameters.x_min[3], max_value=payne_parameters.x_max[3], label_name="vmic"),
        vsini=Parameter(value=3, std=-99, fit=True, min_value=0.0, max_value=100.0, label_name="vsini"),
        vmac=Parameter(value=0, std=-99, fit=False, min_value=0.0, max_value=100.0, label_name="vmac"),
        doppler_shift=Parameter(value=0, std=-99, fit=True, min_value=-5.0, max_value=5.0, label_name="doppler_shift"),
    )

    # add abundances
    for i, label in enumerate(payne_parameters.labels[4:]):
        if label.endswith("_Fe"):
            value_default = 0.0
        else:
            # else is Lithium probably, so we set it to middle value
            value_default = payne_parameters.x_min[i + 4] + (payne_parameters.x_max[i + 4] - payne_parameters.x_min[i + 4]) / 2

        stellar_parameters.abundances[label] = Parameter(
            value=value_default, std=-99, fit=True, min_value=payne_parameters.x_min[i + 4], max_value=payne_parameters.x_max[i + 4], label_name=label
        )

    return stellar_parameters


def fit_stellar_parameters(stellar_parameters: StellarParameters, payne_parameters: PayneParams, wavelength_obs,
                           flux_obs, err_obs, mask_path, do_hydrogen_lines=False, silent=False, fit_carbon=False):
    """
        Fit stellar labels by least-squares matching observed spectra to a Payne model on selected line windows.

        The fit is restricted to pre-defined wavelength windows around diagnostic lines (Mg, Ca, Fe and optionally H).
        This improves robustness versus fitting the full spectrum and targets lines sensitive to Teff/logg/[Fe/H] etc.

        Parameters
        ----------
        stellar_parameters : StellarParameters
            Object holding current parameter values and uncertainties; updated in-place.
        payne_parameters : PayneParams
            Container with Payne coefficients, wavelength grid, labels, and instrumental resolution.
        wavelength_obs, flux_obs, err_obs : array-like
            Observed spectrum and 1-sigma flux uncertainties.
        mask_path : str | Path
            Directory containing CSV masks for line centres (e.g. mg.csv, ca.csv, fe_lines_hr_good.csv, h_cores.csv).
        do_hydrogen_lines : bool, default False
            If True, include hydrogen line cores with wider cut windows.
        silent : bool, default False
            If True, suppress printing.

        Returns
        -------
        StellarParameters
            The input `stellar_parameters` object with fitted values and 1-sigma uncertainties filled in where applicable.

        Notes
        -----
        - Uses `scipy.optimize.curve_fit` on normalised flux (does not renormalise currently).
        - Parameter uncertainties are taken from sqrt(diag(pcov)) (formal covariance; may be optimistic).
        - If all required inputs are already set, fitting is skipped.
        """
    # --- Load line centres (in AA) and define per-line window half-widths (AA) ---
    # The "obs_cut" windows set how much observed spectrum is kept around each line.
    # The "payne_cut" windows set how much of the Payne grid is kept for generating the model.
    # It is bigger because of RV shifts: want to ensure that the model always covers the observed data.
    h_line_cores = pd.read_csv(os.path.join(mask_path, "h_cores.csv"))["ll"]
    h_obs_half_width = [15.0] * len(h_line_cores)
    h_payne_half_width = [20.0] * len(h_line_cores)

    mg_lines = pd.read_csv(os.path.join(mask_path, "mg.csv"))["ll"]
    mg_obs_half_width = [1.0] * len(mg_lines)
    mg_payne_half_width = [1.25] * len(mg_lines)

    ca_lines = pd.read_csv(os.path.join(mask_path, "ca.csv"))["ll"]
    ca_obs_half_width = [1.0] * len(ca_lines)
    ca_payne_half_width = [1.25] * len(ca_lines)

    fe_lines = list(pd.read_csv(os.path.join(mask_path, "fe_lines_hr_good.csv"))["ll"])
    fe_obs_half_width = [1.0] * len(fe_lines)
    fe_payne_half_width = [1.25] * len(fe_lines)

    # --- Assemble the complete line list and matching window sizes ---
    line_centres = list(mg_lines) + list(ca_lines) + list(fe_lines)
    obs_half_widths = mg_obs_half_width + ca_obs_half_width + fe_obs_half_width
    payne_half_widths = mg_payne_half_width + ca_payne_half_width + fe_payne_half_width

    if do_hydrogen_lines:
        line_centres += list(h_line_cores)
        obs_half_widths += h_obs_half_width
        payne_half_widths += h_payne_half_width

    labels_to_fit = ["teff", "logg", "feh", "vmic", "vsini", "vmac", "Mg_Fe", "Ca_Fe", "doppler_shift"]

    if fit_carbon:
        elements = ["c", "o", "ti"]
        for element in elements:
            mask_file = os.path.join(mask_path, f"{element}.csv")

            if os.path.exists(mask_file):
                lines_df = pd.read_csv(mask_file)
                element_lines = list(lines_df["ll"])
                # If per-line dlam is not provided, fall back to a default narrow window.
                dlam = lines_df["dlam"].tolist() if "dlam" in lines_df.columns else [0.5]
            else:
                # if none given, fit basically the whole spectrum
                element_lines = [5000]
                dlam = [3000]

            line_centres += element_lines
            obs_half_widths += dlam
            payne_half_widths += [d * 1.25 for d in dlam]  # scale Payne cut proportionally
        labels_to_fit += ["C_Fe", "O_Fe", "Ti_Fe"]

    # Build the curve_fit inputs: which labels to fit, starting values, and bounds.
    lsq = stellar_parameters.build_lsq_inputs(
        payne_parameters.labels,
        labels_to_fit
    )

    # If all requested labels already have values, don't refit.
    if None not in lsq.input_values:
        if not silent:
            print("All input values are set, skipping fitting.")
        return stellar_parameters

    # Cut observed spectrum and Payne grid down to only the requested line windows.
    (
        wave_obs_cut,
        flux_obs_cut,
        err_obs_cut,
        wave_payne_cut,
        payne_pixel_mask,
    ) = cut_to_just_lines(
        wavelength_obs,
        flux_obs,
        err_obs,
        payne_parameters.wavelength_payne,
        line_centres,
        0,
        obs_cut_aa=obs_half_widths,
        payne_cut_aa=payne_half_widths,
    )

    model_spectrum = make_model_spectrum(
        payne_parameters.payne_coeffs,
        wave_payne_cut,
        resolution_val=payne_parameters.resolution_val,
        pixel_limits=payne_pixel_mask,
        flux_obs=flux_obs_cut,
        refit_continuum=False,
    )

    params = Parameters()
    for label, p0, lo, hi in zip(lsq.labels, lsq.p0, lsq.bounds[0], lsq.bounds[1]):
        params.add(label, value=float(p0), min=float(lo), max=float(hi), vary=True)

    def residual(params, wavelength_obs, flux_obs, err_obs):
        spectra_params = np.array(lsq.input_values, dtype=float)
        j = 0
        for i, val in enumerate(lsq.input_values):
            if val is None:
                spectra_params[i] = params[lsq.labels[j]].value
                j += 1

        model_flux = model_spectrum(wavelength_obs, spectra_params)
        return (model_flux - flux_obs) / err_obs

    if not silent:
        print("Fitting...")

    try:
        result = minimize(
            residual,
            params,
            args=(wave_obs_cut, flux_obs_cut, err_obs_cut),
            method='least_squares',
            max_nfev=1000,
        )
    except (ValueError, RuntimeError) as e:
        print("Error during fitting stellar parameters:", e)
        return stellar_parameters

    if not silent:
        print(fit_report(result))

    for lab in lsq.labels:
        val = result.params[lab].value
        err = result.params[lab].stderr

        if lab in stellar_parameters._core_map:
            attr = stellar_parameters._core_map[lab]
            param = getattr(stellar_parameters, attr)
        else:
            param = stellar_parameters.abundances[lab]

        param.value = float(val)
        param.std = None if err is None else float(err)

    return stellar_parameters

def scale_dlam(dlam, broadening):
    """
        Scale wavelength half-widths used for line fitting as a function of line broadening.

        The scaling is an empirical quadratic relation calibrated for spectra with
        resolving power R ≈ 20 000. For zero broadening the scaling is close to unity
        and increases with increasing broadening to account for wider line profiles.

        Approximate behaviour (for reference):
        - broadening = 0   → scaling ≈ 1
        - broadening = 10  → scaling ≈ 1.2
        - broadening = 50  → scaling ≈ 2.8
        - broadening = 100 → scaling ≈ 5

        Parameters
        ----------
        dlam : float or array-like
            Base wavelength half-width(s) in Å.
        broadening : float
            Broadening parameter in km/s.

        Returns
        -------
        ndarray
            Scaled wavelength half-width(s).
        """

    # Empirical quadratic scaling coefficients (calibrated at R ≈ 20 000)
    a, b, c = 5.7e-05, 0.014481, 0.472108
    scaling = (a * broadening ** 2 + b * broadening + c) * 2.0

    dlam = np.asarray(dlam)

    return dlam * scaling

def fit_one_xfe_element(element_to_fit: str, stellar_parameters: StellarParameters, payne_parameters: PayneParams,
                        wavelength_obs, flux_obs, err_obs, mask_path, silent=False, check_lines_ew=True):
    """
        Fit a single abundance label (e.g. 'Mg_Fe' or 'A_Li') using the Payne model on element-specific line windows.

        The fit is performed only in wavelength windows around the element's diagnostic lines, with window
        half-widths (dlam) typically read from a mask file and expanded empirically based on broadening
        (vsini + vmac). If the fitted line signal is too weak, the result is flagged via errors (becomes -99).

        Parameters
        ----------
        element_to_fit : str
            Payne label to fit (e.g. 'Mg_Fe', 'Ca_Fe', 'A_Li').
        stellar_parameters : StellarParameters
            Current stellar parameters; abundances updated in-place.
        payne_parameters : PayneParams
            Payne coefficients, label ordering, wavelength grid, and label bounds.
        wavelength_obs, flux_obs, err_obs : array-like
            Observed spectrum and 1-sigma flux uncertainties.
        mask_path : str | Path
            Directory containing per-element line masks (CSV with column 'll' and optional 'dlam').
        silent : bool, default False
            If True, suppress printing.
        check_lines_ew : bool, default True
            If True, perform an equivalent-width based sanity check by comparing model spectra with and
            without the fitted element contribution.

        Returns
        -------
        StellarParameters
            Updated `stellar_parameters`.

        Notes
        -----
        - Uses `curve_fit` with provided `sigma`; uncertainties come from sqrt(diag(pcov)).
        - Uses sentinel values (-99) to flag failed/invalid fits.
        - If the best-fit value hits label bounds, it is moved inward and flagged as unreliable.
        """
    # --- Resolve mask file for this element and load line centres / base window widths ---
    if element_to_fit == "A_Li":
        mask_file = os.path.join(mask_path, "li.csv")
    else:
        # Convention: 'Mg_Fe' -> 'mg.csv'
        mask_file = os.path.join(mask_path, f"{element_to_fit.split('_')[0].lower()}.csv")

    if os.path.exists(mask_file):
        lines_df = pd.read_csv(mask_file)
        element_lines = list(lines_df["ll"])
        # If per-line dlam is not provided, fall back to a default narrow window.
        dlam = lines_df["dlam"].tolist() if "dlam" in lines_df.columns else [0.5]
    else:
        # if none given, fit basically the whole spectrum
        element_lines = [5000]
        dlam = [3000]

    # Expand windows to account for rotational + macroturbulent broadening (empirical scaling).
    total_broadening = stellar_parameters.vsini.value + stellar_parameters.vmac.value
    dlam = list(scale_dlam(dlam, total_broadening))

    # Build curve_fit inputs for a one-parameter fit of this element label.
    lsq = stellar_parameters.build_lsq_inputs(payne_parameters.labels, [element_to_fit])

    # If the label is already set, do not refit.
    if None not in lsq.input_values:
        if not silent:
            print("All input values are set, skipping fitting.")
        return stellar_parameters

    # Cut observed spectrum and Payne wavelength grid to just the element windows.
    (
        wave_obs_cut,
        flux_obs_cut,
        err_obs_cut,
        wave_payne_cut,
        payne_pixel_mask,
    ) = cut_to_just_lines(
        wavelength_obs,
        flux_obs,
        err_obs,
        payne_parameters.wavelength_payne,
        element_lines,
        0,
        obs_cut_aa=dlam,
        payne_cut_aa=list(np.asarray(dlam) * 1.5),
    )

    if len(wave_obs_cut) == 0:
        if not silent:
            print(f"No observed data points found in the fitting windows for element {element_to_fit}. Skipping fit.")

        stellar_parameters.abundances[element_to_fit].value = 0
        stellar_parameters.abundances[element_to_fit].std = -99

        return stellar_parameters

    # Build the model-spectrum generator
    model_spectrum = make_model_spectrum(
        payne_parameters.payne_coeffs,
        wave_payne_cut,
        resolution_val=payne_parameters.resolution_val,
        pixel_limits=payne_pixel_mask,
        flux_obs=flux_obs_cut,
        refit_continuum=False,
    )

    # Create lmfit Parameters object for the single fitted label
    params = Parameters()
    params.add(
        element_to_fit,
        value=float(lsq.p0[0]),
        min=float(lsq.bounds[0][0]),
        max=float(lsq.bounds[1][0]),
        vary=True,
    )

    def residual(params, wavelength_obs, flux_obs, err_obs):
        """
        Weighted residuals for lmfit.
        """
        # Rebuild the full spectra_params array from lsq.input_values,
        # replacing the one None with the fitted abundance.
        spectra_params = np.empty(len(lsq.input_values), dtype=float)

        for i, val in enumerate(lsq.input_values):
            if val is None:
                spectra_params[i] = params[element_to_fit].value
            else:
                spectra_params[i] = float(val)

        model_flux = model_spectrum(wavelength_obs, spectra_params)
        return (model_flux - flux_obs) / err_obs

    if not silent:
        print("Fitting...")
        time_start = time.perf_counter()

    try:
        result = minimize(
            residual,
            params,
            args=(wave_obs_cut, flux_obs_cut, err_obs_cut),
            method='least_squares'
        )

        fitted_value = float(result.params[element_to_fit].value)
        stderr = result.params[element_to_fit].stderr
        fitted_error = float(stderr) if stderr is not None else -99.0

        # Optional EW sanity check
        if check_lines_ew:
            idx_element = payne_parameters.labels.index(element_to_fit)

            # Start from current stellar parameters (excluding broadening and RV)
            real_labels = np.array(
                stellar_parameters.build_lsq_inputs(
                    payne_parameters.labels, [element_to_fit]
                ).input_values[:-3],
                dtype=float,
            )

            # Spectrum with fitted element, scaled to Payne
            real_labels[idx_element] = fitted_value
            scaled = (real_labels - payne_parameters.payne_coeffs[-2]) / (
                    payne_parameters.payne_coeffs[-1] - payne_parameters.payne_coeffs[-2]
            ) - 0.5
            spec_with = get_spectrum_from_neural_net(
                scaled_labels=scaled,
                NN_coeffs=payne_parameters.payne_coeffs,
                pixel_limits=payne_pixel_mask,
            )

            # Spectrum with element set to its min value (basically minimum EW that Payne can produce for this stellar parameters)
            real_labels[idx_element] = payne_parameters.payne_coeffs[-2][idx_element]
            scaled = (real_labels - payne_parameters.payne_coeffs[-2]) / (
                    payne_parameters.payne_coeffs[-1] - payne_parameters.payne_coeffs[-2]
            ) - 0.5
            spec_without = get_spectrum_from_neural_net(
                scaled_labels=scaled,
                NN_coeffs=payne_parameters.payne_coeffs,
                pixel_limits=payne_pixel_mask,
            )

            # Ensure one dlam value per each line for the integration windows.
            if len(dlam) == 1:
                dlam_per_line = [dlam[0]] * len(element_lines)
            else:
                dlam_per_line = dlam

            ew_deltas = []
            for line, dlam_one in zip(element_lines, dlam_per_line):
                try:
                    ew_with = calculate_equivalent_width(
                        wave_payne_cut, spec_with, line - dlam_one, line + dlam_one
                    )
                    ew_without = calculate_equivalent_width(
                        wave_payne_cut, spec_without, line - dlam_one, line + dlam_one
                    )
                    ew_deltas.append(ew_with - ew_without)
                except ValueError:
                    if not silent:
                        print(f"EW check failed for element {element_to_fit}")
                    continue

            if ew_deltas:
                avg_ew = float(np.mean(ew_deltas))
                max_ew = float(np.max(ew_deltas))
                # Thresholds are in mÅ
                if avg_ew < 1 and max_ew < 2:
                    if not silent:
                        print(
                            f"Equivalent width of {element_to_fit} is too low: "
                            f"{avg_ew:.2f}, max={max_ew:.2f}"
                        )
                    # Keep value but flag uncertainty as invalid/unreliable.
                    fitted_error = -99.0

    except (ValueError, RuntimeError, TypeError):
        if not silent:
            print(f"Fitting failed for element {element_to_fit}")
        fitted_value, fitted_error = -99.0, -99.0

    # --- Post-processing: handle failed fits and boundary-hitting solutions ---
    idx = payne_parameters.labels.index(element_to_fit)

    if fitted_value < -90:
        # Normalise failed fits to 0 but keep sentinel error.
        # this is because Payne will produce nonsense if the fitted value is not in the range
        fitted_value, fitted_error = 0.0, -99.0
    elif np.abs(fitted_value - payne_parameters.x_max[idx]) <= 0.02:
        # If the fit pegs at bounds, move inward and flag as unreliable.
        # TODO: sometimes this makes sense, sometimes it is a min/max bound. But often means the line is too weak and
        #  the fit is not sure what value to choose - too much noise dependent.
        fitted_value = payne_parameters.x_max[idx] - (
                payne_parameters.x_max[idx] - payne_parameters.x_min[idx]
        ) / 3.0
        fitted_error = -99.0
    elif np.abs(fitted_value - payne_parameters.x_min[idx]) <= 0.02:
        # If the fit pegs at bounds, move inward and flag as unreliable.
        fitted_value = payne_parameters.x_min[idx] + (
                payne_parameters.x_max[idx] - payne_parameters.x_min[idx]
        ) / 3.0
        fitted_error = -99.0

    if not silent:
        print(f"Done fitting in {time.perf_counter() - time_start:.2f} seconds")
        print(f"Fitted {element_to_fit}: {fitted_value:.3f} +/- {fitted_error:.3f}")

    # Write result back into the abundance container if it exists.
    if element_to_fit in stellar_parameters.abundances:
        stellar_parameters.abundances[element_to_fit].value = fitted_value
        stellar_parameters.abundances[element_to_fit].std = fitted_error

    return stellar_parameters


def plot_fitted_payne(wavelength_payne, final_parameters, payne_coeffs, wavelength_obs, flux_obs, labels, resolution_val=None, plot_show=True):
    doppler_shift = final_parameters['doppler_shift']
    vmac = final_parameters['vmac']
    vsini = final_parameters['vsini']

    final_params = []

    for label in labels:
        final_params.append(final_parameters[label])

    real_labels = final_params

    scaled_labels = (real_labels - payne_coeffs[-2]) / (payne_coeffs[-1] - payne_coeffs[-2]) - 0.5
    payne_fitted_spectra = get_spectrum_from_neural_net(scaled_labels=scaled_labels,
                                                                       NN_coeffs=payne_coeffs)

    wavelength_payne_plot = wavelength_payne
    if vmac > 1e-3:
        wavelength_payne_plot, payne_fitted_spectra = conv_macroturbulence(wavelength_payne_plot,
                                                                           payne_fitted_spectra, vmac)
    if vsini > 1e-3:
        wavelength_payne_plot, payne_fitted_spectra = conv_rotation(wavelength_payne_plot, payne_fitted_spectra,
                                                                    vsini)
    if resolution_val is not None:
        wavelength_payne_plot, payne_fitted_spectra = conv_res(wavelength_payne_plot, payne_fitted_spectra,
                                                               resolution_val)

    # cut wavelength_obs to the payne windows
    windows = [[3926, 4355], [5160, 5730], [6100, 6790]]
    wavelength_obs_cut = []
    flux_obs_cut = []
    for window in windows:
        mask = (wavelength_obs > window[0]) & (wavelength_obs < window[1])
        wavelength_obs_cut.extend(wavelength_obs[mask])
        flux_obs_cut.extend(flux_obs[mask])

    plt.figure(figsize=(18, 6))
    plt.scatter(wavelength_obs_cut, flux_obs_cut, label="Observed", s=3, color='k')
    plt.plot(wavelength_payne_plot * (1 + (doppler_shift / 299792.)), payne_fitted_spectra, label="Payne",
             color='r')
    plt.legend()
    plt.ylim(0.0, 1.05)
    plt.xlim(wavelength_payne_plot[0], wavelength_payne_plot[-1])
    if plot_show:
        plt.show()
