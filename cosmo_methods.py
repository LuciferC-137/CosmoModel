import numpy as np
from logger import Logger
from utils import Units
from scipy.integrate import solve_ivp, quad
from scipy.integrate import cumulative_trapezoid

# ---------------------------- SOLVER METHODS -------------------------


def z(a, t: np.ndarray) -> np.ndarray:
    """Redshift. Parameter 'a' can be a function or an array."""
    if isinstance(a, float) or isinstance(a, np.ndarray) or isinstance(a, int):
        return Units.a0 / a - 1
    return Units.a0 / a(t) - 1

def E(z: np.ndarray) -> np.ndarray:
    """Dimensionless Hubble parameter."""
    return np.sqrt(Units.Omega_r * np.power(1 + z, 4) 
                   + Units.Omega_m * np.power(1 + z, 3)
                   + Units.Omega_lambda)

def friedmann(t: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Define the differential equation on a(t) to solve Friedmann's equation.
    """
    dadt = Units.H0_per_Gyr * a * E(z(a, t))
    return dadt

def log_friedmann(logt: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Define the differential equation on a logarithmic time scale
    to solve Friedmann's equation.
    """
    dadt = Units.H0_per_Gyr * np.log(10) * a * np.power(10, logt) \
        * E(z(a, 10**logt))
    return dadt

def a_analytic(tGy: np.ndarray) -> np.ndarray:
    """
    This is the analytic solution of the Friedmann equation for a universe with
    matter and lambda:
    """
    if (Units.Omega_m == 1.):
        return Units.a0 * np.power(tGy / Units.today, 2/3)
    a_origin = Units.a0 * np.power(tGy[0] / Units.today, 2/3)
    t = tGy * Units.sec_in_Gyr
    A = np.power(np.sinh(np.sqrt(3 * Units.Lambda / 4) * t[0]), -2/3) * a_origin
    return A * np.power(np.sinh(np.sqrt(3 * Units.Lambda / 4) * t), 2/3)

def a_solve(t: np.ndarray, ainit: float, log: bool = False) -> np.ndarray:
    """Scale factor."""
    if log:
        return solve_ivp(friedmann, (t[0], t[-1]), [ainit], t_eval=t)
    return solve_ivp(log_friedmann, (t[0], t[-1]), [ainit], t_eval=t)

# ------------------------ REGULAR TIME ------------------------

def particule_horizon(a_large: np.ndarray, time: np.ndarray,
                      large_time: np.ndarray) -> np.ndarray:
    """Particle horizon comoving distance. (Glyr)"""
    p_h = Units.a0 * Units.c_Glyr_per_Gyr \
        * cumulative_trapezoid(1 / a_large,
                               large_time, initial=0)
    p_h_truncated = p_h[large_time <= time[-1]]
    return np.interp(time, large_time[:len(p_h_truncated)], p_h_truncated)

def event_horizon(a_large: np.ndarray, time: np.ndarray,
                  large_time: np.ndarray = None) -> np.ndarray:
    """Event horizon comoving distance. (Glyr)"""
    e_h = Units.a0 * Units.c_Glyr_per_Gyr \
        * cumulative_trapezoid(1 / a_large[::-1],
                               large_time[::-1], initial=0)[::-1]
    e_h_truncated = e_h[large_time <= time[-1]]
    return np.interp(time, large_time[:len(e_h_truncated)], e_h_truncated)

def hubble_sphere(a: np.ndarray, t: float) -> float:
    """Hubble sphere comoving distance. (Glyr)"""
    if isinstance(a, np.ndarray) or isinstance(a, float) or isinstance(a, int):
        return Units.a0 * Units.c_Glyr_per_Gyr / friedmann(t, a)
    return Units.a0 * Units.c_Glyr_per_Gyr / friedmann(t, a(t))

def light_cone(a: np.ndarray, tem: np.ndarray) -> np.ndarray:
    """Light cone comoving distance. (Glyr)"""
    restricted_time = tem[tem < Units.today]
    # Reversing a for integration from t to tmax
    a_inv = 1 / a[tem < Units.today][::-1]
    l_c = Units.a0 * Units.c_Glyr_per_Gyr \
        * cumulative_trapezoid(a_inv, restricted_time, initial=0)[::-1]
    return np.concatenate((l_c, np.ones(len(tem)
                                        - len(restricted_time)) * np.nan))

# ------------------------ CONFORMAL TIME ------------------------

def conformal_time(a: callable, t: float, tmin: float) -> float:
    """Conformal time. (Gyr)"""
    if isinstance(a, np.ndarray) or isinstance(a, float) or isinstance(a, int):
        Logger().raise_error("Use conformal_time_vect for arrays.")
    return Units.a0 * quad(lambda x: 1 / a(x), tmin, t)[0]

def particle_horizon_conformal(t: float, tmin: float = 0) -> float:
    """Particle horizon comoving distance. (Glyr)"""
    return Units.c_Glyr_per_Gyr * (t - tmin)

def event_horizon_conformal(t: float, t_max: float) -> float:
    """Event horizon comoving distance. (Glyr)"""
    return Units.c_Glyr_per_Gyr * (t_max - t)

def light_cone_conformal(today_conformal: float, tem: float) -> float:
    """Light cone comoving distance. (Glyr)"""
    if (tem > today_conformal):
        return np.nan
    return Units.c_Glyr_per_Gyr * (today_conformal - tem)

# ---------------- CONVERTING REDSHIFT TO DISTANCE ----------------

def chi_from_z(z: float) -> float:
    """Comoving distance from redshift z to today (Glyr)."""
    return Units.c_Glyr_per_Gyr / Units.H0_per_Gyr / Units.a0 \
            * quad(lambda x: 1. / E(x), 0, z)[0]

def iso_chi(a: callable, t: np.ndarray, chi: float) -> np.ndarray:
    """Isochrone redshift. (Glyr)"""
    return a(t) * chi

# ------------------ OPTIMIZED METHODS ------------------------

def conformal_time_vect(a: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Optimized conformal time using cumulative integration."""
    inv_a = 1 / a
    integral = cumulative_trapezoid(inv_a, t, initial=0)
    return Units.a0 * integral
