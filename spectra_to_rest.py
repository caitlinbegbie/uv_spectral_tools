from astropy.io import fits
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.ndimage import maximum_filter1d
from sklearn.model_selection import KFold
from scipy.interpolate import interp1d
from cross_correlate import cross_correlate
import chart_studio.plotly as py
import plotly.graph_objects as go
import cufflinks as cf
import seaborn as sns
import plotly.express as px

#### ---------------- MAIN FUNCTION ---------------- ####

def spectra_to_rest(file_path):
    """
    Overall function to shift a file into analysis-ready format.

    Input:
        * HASP .fits file
    Outputs:
        * rest-shifted wavelengths
        * continuum - normalized fluxes
    """
    aca_basis = 'data/shifting_data/hst_14788_stis_hd128620_e230h_od5c_cspec.fits'   # chosen by SNR values

    # Step 1: load input spectrum and basis spectrum (internal)
    wl_a, flux_a = load_spectrum(aca_basis)
    wl_b, flux_b = load_spectrum(file_path)

    # Step 2: Define envelope
    env_wl_a, env_flux_a = define_envelope(wl_a, flux_a)
    env_wl_b, env_flux_b = define_envelope(wl_b, flux_b)

    # Step 3: Fit envelope using Gaussian processes
    cont_a, _ = fit_continuum(wl_a, env_wl_a, env_flux_a)
    cont_b, _ = fit_continuum(wl_b, env_wl_b, env_flux_b)
    
    # Step 4: normalize files with room for error
    try:
        flux_norm_a = normalize_spectrum(wl_a, flux_a, cont_a)
        flux_norm_b = normalize_spectrum(wl_b, flux_b, cont_b)
        
    except ValueError as e:
        print(f'Cannot normalize {file_path}: {e}')
        return None, None
    
    # Step 5: create common wl grid to ensure length of each spectra is the same 
    wl_min = max(wl_a.min(), wl_b.min())
    wl_max = min(wl_a.max(), wl_b.max())

    if wl_max <= wl_min:
        print(f'Cannot analyze {file_path} - wavelength ranges do not overlap with basis!')
        return None, None
        
    common_wl = np.linspace(max(wl_a.min(), wl_b.min()), min(wl_a.max(), wl_b.max()), min(len(wl_a), len(wl_b)))

    # Step 6: interpolate normalized continuum fluxes to common wavelength grid
    interp_a = interp1d(wl_a, flux_norm_a, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_b = interp1d(wl_b, flux_norm_b, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    new_flux_norm_a = interp_a(common_wl)
    new_flux_norm_b = interp_b(common_wl)

    # checking for invalid/empty results
    if len(new_flux_norm_a) == 0 or len(new_flux_norm_b) == 0:
        print(f"Cannot analyze {file_path} — interpolated flux array is empty.")
        return None, None

    if np.all(np.isnan(new_flux_norm_a)) or np.all(np.isnan(new_flux_norm_b)):
        print(f"Cannot analyze {file_path} — interpolated flux is all NaNs.")
        return None, None
    
    # Step 6: cross correlate to basis spectrum
    shift, corr_coeff, err = cross_correlate(new_flux_norm_a, new_flux_norm_b, wave_a=common_wl, wave_b=common_wl)

    # checking that error < correlation coefficient
    if (err >= corr_coeff):
        print(f'Cannot cross-correlate {file_path}: Error exceeds correlation coefficient!')
        return None, None

    # Step 7: Apply wavelength shift of original input spectra using Alpha Cen A RV value
    shifted_wls = wl_b + shift
    rv_rest_wls = rv_to_rest(shifted_wls)

    # Step 8: Convert radially shifted vaccuum wavelengths to air wavelengths 
    rv_rest_air = air_conversion(rv_rest_wls)

    # Step 9: Apply last minimal air shift using average offset from FeI rest lines
    rest_wls = feii_shift(rv_rest_air)

    return rest_wls, flux_norm_b



#### ---------------- SHIFTING TO ALPHA CEN WLS ---------------- ####
def shift_to_aca(file, plot=False):
    """
    Finds the shift from an input file (HASP spectral product) to the Alpha Centauri A RV frame. 
    Further shift by Alpha Centauri A RV is needed to shift the spectrum into rest wavelengths.
    Outputs: wavelength shift (in Å), correlation coefficient, error

    if plot=True, function will produce a plot of the original Alpha Centauri A spectrum compared to the shifted comparison spectrum using matplotlib.
    """
    # Load basis and target spectra
    aca_basis = 'data/shifting_data/hst_14788_stis_hd128620_e230h_od5c_cspec.fits'   # chosen by SNR values
    star_name = file.split('_')[3]
    
    wl_a, flux_a = load_spectrum(aca_basis)
    wl_b, flux_b = load_spectrum(file)

    env_wl_a, env_flux_a = define_envelope(wl_a, flux_a)
    env_wl_b, env_flux_b = define_envelope(wl_b, flux_b)

    cont_a, _ = fit_continuum(wl_a, env_wl_a, env_flux_a)
    cont_b, _ = fit_continuum(wl_b, env_wl_b, env_flux_b)
    
    # normalize files with room for error
    try:
        flux_norm_a = normalize_spectrum(wl_a, flux_a, cont_a)
        flux_norm_b = normalize_spectrum(wl_b, flux_b, cont_b)
        
    except ValueError as e:
        print(f'Skipping {file}: {e}')
        return None, None, None
    
    # create common wl grid to ensure length of each spectra is the same 
    wl_min = max(wl_a.min(), wl_b.min())
    wl_max = min(wl_a.max(), wl_b.max())

    if wl_max <= wl_min:
        print(f'Skipping {file} - wavelength ranges do not overlap!')
        return None, None, None
        
    common_wl = np.linspace(max(wl_a.min(), wl_b.min()), min(wl_a.max(), wl_b.max()), min(len(wl_a), len(wl_b)))
    
    # interpolating normalized continuum fluxes to common grid
    interp_a = interp1d(wl_a, flux_norm_a, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_b = interp1d(wl_b, flux_norm_b, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    new_flux_norm_a = interp_a(common_wl)
    new_flux_norm_b = interp_b(common_wl)

    # checking for invalid/empty results
    if len(new_flux_norm_a) == 0 or len(new_flux_norm_b) == 0:
        print(f"Skipping {file} — interpolated flux array is empty.")
        return None, None, None

    if np.all(np.isnan(new_flux_norm_a)) or np.all(np.isnan(new_flux_norm_b)):
        print(f"Skipping {file} — interpolated flux is all NaNs.")
        return None, None, None
    
    
    # cross correlate
    shift, corr_coeff, err = cross_correlate(new_flux_norm_a, new_flux_norm_b, wave_a=common_wl, wave_b=common_wl)

    # checking that error < correlation coefficient
    if (err >= corr_coeff):
        print(f'Skipping {file}: Error exceeds correlation coefficient!')
        return None, None, None

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(wl_a, flux_norm_a, color='blue', label='Alpha Cen A Basis Spectrum')
        plt.plot(wl_b + shift, flux_norm_b, color='pink', label=f'{star_name} Spectrum (Shifted)', alpha=0.8)
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Flux')
        plt.title(f'Cross-Correlation of {star_name} Spectra to Alpha Centauri A')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return shift, corr_coeff, err



#### ---------------- DEPENDENCY FUNCTIONS ---------------- ####

def load_spectrum(file_path):
    """
    Loads in a HASP product .fits file and cleans data for envelope and continuum fitting.
    
    Filters omit spectral considerations due to:
        * nans
        * infs

    Returns spectral wavelengths and fluxes filtered along these guidelines.
    """
    with fits.open(file_path) as hdul:
        data = hdul[1].data
        wl = data['WAVELENGTH']
        flux = data['FLUX']

    mask = (np.isfinite(wl)) & (np.isfinite(flux))
    wl_clean = wl[mask]
    flux_clean = flux[mask]

    return wl_clean, flux_clean


def define_envelope(wl, flux, window=200):
    """
    Identifies the upper envelope of a spectrum using a filter that finds the local maximum for each window.
    Designed to omit large spectra deviations from continuum before fitting by tossing out flux values below local maximum. 
    Default window size is 100 but changeable based on effectivity.
    """
    
    std_flux = np.std(flux)
    mask = (
        (flux > 0) &
        ((wl < 2788) | (wl > 2805)) &
        (flux < 5 * std_flux)
    )

    fit_wl = wl[mask]
    fit_flux = flux[mask]
    max_flux = maximum_filter1d(fit_flux, size=window)
    mask = np.where(fit_flux >= max_flux)
    return fit_wl[mask], fit_flux[mask]


def fit_continuum(star_wl, env_wl, env_flux):
    """
    Fits the envelope outline given by define_envelope using sklearn's Matern kernel in GaussianProcessRegressor. 
    Matern kernel relies on two parameters (nu and length_scale) -- optimal parameters are found through chi-squared analysis.
    Returns fluxes of spectra continuum.
    """
    # creating input array (wavelength) amd output array (flux)
    X = env_wl.reshape(-1, 1)
    y = env_flux
    x = np.linspace(np.min(env_wl), np.max(env_wl), len(env_wl)).reshape(-1, 1)
    
    # Define analysis grids for parameter optimization
    length_scales = np.logspace(2, 4, 100)   # tests length scales from ~100 - 10000 Å
    nus = [0.5, 1.5, 2.5]   # only ones really allowed by Matern

    # Constructing chi-squared map tp find best fit
    chi2_map = np.zeros((len(nus), len(length_scales)))
    N_data = len(y)
    N_params = 2
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    chi2_map = np.zeros((len(nus), len(length_scales)))
    
    for i, nu in enumerate(nus):
        for j, ls in enumerate(length_scales):
            kernel = Matern(length_scale=ls, nu=nu)
            gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True)
    
            residuals = []
            try:
                for train_idx, test_idx in kf.split(X):
                    gp.fit(X[train_idx], y[train_idx])
                    y_pred = gp.predict(X[test_idx])
                    residuals.append((y[test_idx] - y_pred)**2)
    
                residuals = np.concatenate(residuals)
                chi2 = np.median(residuals)
                chi2_red = chi2 / (N_data - N_params)  # Reduced chi^2
                chi2_map[i, j] = np.log10(chi2_red)
            except Exception as e:
                chi2_map[i, j] = np.nan

    # Find best fit from chi2 map
    min_idx = np.unravel_index(np.nanargmin(chi2_map), chi2_map.shape)
    best_nu = nus[min_idx[0]]
    best_ls = length_scales[min_idx[1]]
    
    # Fit envelope using derived parameters
    best_kernel = Matern(length_scale=best_ls, length_scale_bounds="fixed", nu=best_nu)
    gp_best = GaussianProcessRegressor(kernel=best_kernel, normalize_y=True)
    gp_best.fit(X, y)
    full_wls = star_wl.reshape(-1, 1)
    continuum, std = gp_best.predict(full_wls, return_std=True)

    return continuum, std


def normalize_spectrum(wl, flux, continuum):
    """
    Designed to normalize spectra using median value of characteristically flat region from ~1650 - 1700 Å in Sun-like stars.
    Unsure yet if this process will work in chemically peculiar instances.
    """
    # normalize using median of flat region (trending feature)
    flat_idx = np.where((wl > 2650) & (wl < 2700))

    if flat_idx[0].size == 0:
        raise ValueError(f"No points found in flat region 2650–2700 Å for normalization.")
    
    flat_wls, flat_fluxes = wl[flat_idx], continuum[flat_idx]
    norm_val = np.median(flat_fluxes)

    norm_fluxes = flux / norm_val

    return norm_fluxes


def normalize_spectrum_from_file(file):
    """
    Conglomeration of previous functions to allow normalization from a filepath.
    """  
    wl_init, flux_init = load_spectrum(file)
    env_wl, env_flux = define_envelope(wl_init, flux_init)
    continuum, std = fit_continuum(wl_init, env_wl, env_flux)
    norm_fluxes = normalize_spectrum(wl_init, flux_init, continuum)

    return norm_fluxes


def rv_to_rest(wl):
    rv = -22.4   #median value from SIMBAD (km/s)
    c = 2.998e5   # km/s
    rest_wl = wl / ((rv/c) + 1)
    return rest_wl


def air_conversion(wave):
    '''
    Converts a vacuum wavelength spectrum to air following Shetrone et al.
    (2015).
    '''

    a = 0.0
    b1 = 5.792105e-2
    b2 = 1.67917e-3
    c1 = 238.0185
    c2 = 57.362

    wave = wave / 10000.

    air_conv = a + (b1 / (c1 - 1 / (wave**2.))) + \
        (b2 / (c2 - 1 / (wave**2.))) + 1

    wave_air = wave / air_conv
    wave_air = wave_air * 10000.

    return wave_air

def feii_shift(air_wavelength):
    """
    Final calculated shift calculated by taking the average offset of rest air wavelengths
    to 23 accepted FeI line positions.
    """
    avg_feii_shift = 0.02328
    return air_wavelength - avg_feii_shift