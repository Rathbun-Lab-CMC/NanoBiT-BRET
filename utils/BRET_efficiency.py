import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Optional
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

def lognormal_function(
    nu: np.ndarray, 
    Im: float, 
    nu_m: float, 
    nu_minus: float, 
    nu_plus: float
) -> np.ndarray:
    """
    Calculate log-normal function for emission spectra.
    
    Equation taken from: https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1751-1097.1996.tb02464.x
    I(ν) = Im * exp{ - (ln 2 / ln²(ρ)) * ln²[ (a - ν)/(a - ν_m) ] }, if ν < a
         = 0 otherwise
    
    Parameters:
    -----------
    nu : np.ndarray
        Wavenumber values (cm⁻¹)
    Im : float
        Maximum intensity
    nu_m : float
        Wavenumber at maximum intensity
    nu_minus : float
        Lower wavenumber bound
    nu_plus : float
        Upper wavenumber bound
    
    Returns:
    --------
    np.ndarray
        Intensity values at each wavenumber
    """
    nu = np.asarray(nu)
    H = nu_plus - nu_minus
    rho = (nu_m - nu_minus) / (nu_plus - nu_m)

    # Prevent division by zero or invalid log if rho ≈ 1
    if np.isclose(rho, 1.0):
        rho += 1e-5

    a = nu_m + H * rho / (rho**2 - 1)
    output = np.zeros_like(nu)


    valid = nu < a

    ln2 = np.log(2)
    denom = (np.log(rho)) ** 2
    ratio = (a - nu[valid]) / (a - nu_m)
    output[valid] = Im * np.exp(-(ln2 / denom) * (np.log(ratio) ** 2))
    
    return output


def wavelength_to_wavenumber(
    wavelength_nm: np.ndarray, 
    intensity: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert wavelength data to wavenumber with intensity correction.
    
    Parameters:
    -----------
    wavelength_nm : np.ndarray
        Wavelength values in nanometers
    intensity : np.ndarray
        Intensity values
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Wavenumber (cm⁻¹) and corrected intensity arrays, sorted by wavenumber
    """
    wavenumber_cm1 = 1e7 / wavelength_nm
    corrected_intensity = intensity * (wavelength_nm ** 2)
    sort_idx = np.argsort(wavenumber_cm1)
    
    return wavenumber_cm1[sort_idx], corrected_intensity[sort_idx]


def fit_lognormal(
    nu: np.ndarray, 
    I_nu: np.ndarray, 
    fit_range: Optional[Tuple[float, float]] = None
) -> Tuple[float, float, float, float, float]:
    """
    Fit log-normal function to emission data.
    
    Parameters:
    -----------
    nu : np.ndarray
        Wavenumber values (cm⁻¹)
    I_nu : np.ndarray
        Intensity values
    fit_range : Optional[Tuple[float, float]], optional
        Range of wavenumbers to fit (min, max). If None, fit entire range.
    
    Returns:
    --------
    Tuple[float, float, float, float, float]
        Fitted parameters: Im, nu_m, nu_minus, nu_plus, r_squared
    
    Raises:
    -------
    RuntimeError
        If parameter estimation fails
    """
    if fit_range is not None:
        mask = (nu >= fit_range[0]) & (nu <= fit_range[1])
        nu_fit = nu[mask]
        I_fit = I_nu[mask]
    else:
        nu_fit = nu
        I_fit = I_nu

    Im_guess = I_fit.max()
    nu_m_guess = nu_fit[np.argmax(I_fit)]

    half_max = Im_guess / 2
    try:
        nu_minus_guess = nu_fit[nu_fit < nu_m_guess][
            np.argmin(np.abs(I_fit[nu_fit < nu_m_guess] - half_max))
        ]
        nu_plus_guess = nu_fit[nu_fit > nu_m_guess][
            np.argmin(np.abs(I_fit[nu_fit > nu_m_guess] - half_max))
        ]
    except Exception as e:
        raise RuntimeError(f"Failed to estimate ν₋ and ν₊: {e}")

    p0 = [Im_guess, nu_m_guess, nu_minus_guess, nu_plus_guess]

    bounds = (
        [0, nu_fit.min(), nu_fit.min(), nu_fit.min()],
        [np.inf, nu_fit.max(), nu_fit.max(), nu_fit.max()]
    )

    popt, _ = curve_fit(
        lognormal_function, 
        nu_fit, 
        I_fit, 
        p0=p0, 
        bounds=bounds, 
        maxfev=20000
    )

    Im, nu_m, nu_minus, nu_plus = popt
    fit_y = lognormal_function(nu_fit, *popt)
    r2 = 1 - np.sum((I_fit - fit_y) ** 2) / np.sum((I_fit - np.mean(I_fit)) ** 2)

    return Im, nu_m, nu_minus, nu_plus, r2


def extrapolate_and_fit_lognormal(
    wavelengths: np.ndarray,
    emission: np.ndarray,
    fit_range: Optional[Tuple[float, float]] = None,
    extrapolate_range: Tuple[float, float] = (300, 700),
    num_points: int = 220
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit and extrapolate emission using log-normal function.
    
    Parameters:
    -----------
    wavelengths : np.ndarray
        Wavelength values in nanometers
    emission : np.ndarray
        Emission intensity values
    fit_range : Optional[Tuple[float, float]], optional
        Wavelength range for fitting (min, max). If None, fit entire range.
    extrapolate_range : Tuple[float, float], optional
        Wavelength range for extrapolation (min, max), default (300, 700)
    num_points : int, optional
        Number of points in extrapolated data, default 220
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Extrapolated wavelengths and intensities
    """
    nu, I_nu = wavelength_to_wavenumber(wavelengths, emission)
    
    if fit_range is not None:
        fit_range_nu = (1e7 / fit_range[1], 1e7 / fit_range[0])  # invert wavelength order
    else:
        fit_range_nu = None

    Im, nu_m, nu_minus, nu_plus, r2 = fit_lognormal(nu, I_nu, fit_range=fit_range_nu)

    # Generate extrapolated wavelengths (nm), then convert to wavenumbers
    wavelengths_extrap = np.linspace(*extrapolate_range, num_points)
    nu_extrap = 1e7 / wavelengths_extrap
    I_nu_extrap = lognormal_function(nu_extrap, Im, nu_m, nu_minus, nu_plus)

    # Convert back to wavelength space
    I_lambda_extrap = I_nu_extrap / (wavelengths_extrap ** 2)

    return wavelengths_extrap, I_lambda_extrap


def gaussian_function(
    x: np.ndarray, 
    amplitude: float, 
    center: float, 
    sigma: float
) -> np.ndarray:
    """
    Calculate Gaussian function.
    
    Parameters:
    -----------
    x : np.ndarray
        Input values
    amplitude : float
        Amplitude of the Gaussian
    center : float
        Center position of the Gaussian
    sigma : float
        Standard deviation of the Gaussian
    
    Returns:
    --------
    np.ndarray
        Gaussian function values
    """
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))


def fit_gaussian(
    wavelengths: np.ndarray,
    emission: np.ndarray,
    fit_range: Optional[Tuple[float, float]] = None
) -> Tuple[float, float, float, float]:
    """
    Fit Gaussian function to emission data.
    
    Parameters:
    -----------
    wavelengths : np.ndarray
        Wavelength values in nanometers
    emission : np.ndarray
        Emission intensity values
    fit_range : Optional[Tuple[float, float]], optional
        Wavelength range for fitting (min, max). If None, fit entire range.
    
    Returns:
    --------
    Tuple[float, float, float, float]
        Fitted parameters: amplitude, center, sigma, r_squared
    """
    if fit_range is not None:
        mask = (wavelengths >= fit_range[0]) & (wavelengths <= fit_range[1])
        wavelengths_fit = wavelengths[mask]
        emission_fit = emission[mask]
    else:
        wavelengths_fit = wavelengths
        emission_fit = emission

    amplitude_guess = np.max(emission_fit)
    center_guess = wavelengths_fit[np.argmax(emission_fit)]
    sigma_guess = 5

    p0 = [amplitude_guess, center_guess, sigma_guess]

    popt, _ = curve_fit(gaussian_function, wavelengths_fit, emission_fit, p0=p0, maxfev=10000)
    fitted_amplitude, fitted_center, fitted_sigma = popt

    fitted_emission = gaussian_function(wavelengths_fit, *popt)
    ss_res = np.sum((emission_fit - fitted_emission) ** 2)
    ss_tot = np.sum((emission_fit - np.mean(emission_fit)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return fitted_amplitude, fitted_center, fitted_sigma, r_squared


def extrapolate_and_fit_gaussian(
    wavelengths: np.ndarray,
    emission: np.ndarray,
    fit_range: Optional[Tuple[float, float]] = None,
    extrapolate_range: Tuple[float, float] = (300, 700),
    num_points: int = 220
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit and extrapolate emission using Gaussian function.
    
    Parameters:
    -----------
    wavelengths : np.ndarray
        Wavelength values in nanometers
    emission : np.ndarray
        Emission intensity values
    fit_range : Optional[Tuple[float, float]], optional
        Wavelength range for fitting (min, max). If None, fit entire range.
    extrapolate_range : Tuple[float, float], optional
        Wavelength range for extrapolation (min, max), default (300, 700)
    num_points : int, optional
        Number of points in extrapolated data, default 220
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Extrapolated wavelengths and intensities
    """
    amplitude, center, sigma, r2 = fit_gaussian(wavelengths, emission, fit_range)

    x_extrap = np.linspace(extrapolate_range[0], extrapolate_range[1], num_points)
    y_extrap = gaussian_function(x_extrap, amplitude, center, sigma)

    return x_extrap, y_extrap

def fit_spectrum(wavelengths, emission, fit_range=None):
    try:
        # Try log-normal fit first
        x, y = extrapolate_and_fit_lognormal(wavelengths, emission, fit_range=fit_range)
        
        # Check if the fit produced valid results (not all zeros)
        if np.all(y == 0) or np.any(np.isnan(y)) or np.any(np.isinf(y)):
            # Fallback to Gaussian fit
            x, y = extrapolate_and_fit_gaussian(wavelengths, emission, fit_range=fit_range)
            
    except (RuntimeError, ValueError, TypeError):
        # If log-normal fit fails, use Gaussian as fallback
        x, y = extrapolate_and_fit_gaussian(wavelengths, emission, fit_range=fit_range)
    
    return x, y


def compute_estimated_AUC_spectrum(wavelengths, emission, fit_func, full_range=(300, 700), num_points=400):
    # Interpolate raw
    interp_data = interp1d(wavelengths, emission, bounds_error=False, fill_value=0)
    # Get extrapolated fit
    x_fit = np.linspace(full_range[0], full_range[1], num_points)
    y_fit = fit_func(x_fit)
    
    y_combined = np.where((x_fit >= wavelengths[0]) & (x_fit <= wavelengths[-1]),
                          interp_data(x_fit),
                          y_fit)
    # Integrate full spectrum
    auc = trapezoid(y_combined, x_fit)
    return auc, x_fit, y_combined


