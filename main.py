import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d
from utils import DataHolder, Units, FilteredOutput
from plotter import Plotter

# Filtering warnings for a cleaner output. When this is on, 
# all print() functions must explicitly end with '\n' to make a new line.
FilteredOutput.on()

# ---------------------------- SOLVER METHODS -------------------------

def z(a: callable, t: np.ndarray) -> np.ndarray:
    """Redshift. Parameter 'a' can be a function or an array."""
    if isinstance(a, float) or isinstance(a, np.ndarray):
        return Units.a0 / a - 1
    return Units.a0 / a(t) - 1

def E(z: np.ndarray) -> np.ndarray:
    """Dimensionless Hubble parameter."""
    return np.sqrt(Units.Omega_m * (1 + z)**3 + Units.Omega_lambda)

def friedmann(t: float, a: float) -> float:
    """
    Define the differential equation on a(t) to solve Friedmann's equation.
    """
    dadt = Units.H0_per_Gyr * a * E(z(a, t))
    return dadt

def log_friedmann(logt: float, a: float) -> float:
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

def particule_horizon(a:callable, t: float, tmin: float = 0) -> float:
    """Particle horizon comoving distance. (Glyr)"""
    return Units.a0 * Units.c_Glyr_per_Gyr \
        * quad(lambda x: 1 / a(x), tmin, t)[0]

def event_horizon(a:callable, t: float, t_max: float) -> float:
    """Event horizon comoving distance. (Glyr)"""
    return Units.a0 * Units.c_Glyr_per_Gyr \
        * quad(lambda x: 1 / a(x), t, t_max)[0]

def hubble_sphere(a:callable, t: float) -> float:
    """Hubble sphere comoving distance. (Glyr)"""
    return Units.a0 * Units.c_Glyr_per_Gyr / friedmann(t, a(t))

def light_cone(a:callable, tem: float) -> float:
    """Light cone comoving distance. (Glyr)"""
    if (tem > Units.today):
        return np.nan
    return Units.a0 * Units.c_Glyr_per_Gyr \
        * quad(lambda x: 1 / a(x), tem, Units.today)[0]

# ------------------------ CONFORMAL TIME ------------------------

def conformal_time(a: callable, t: float, tmin: float) -> float:
    """Conformal time. (Gyr)"""
    return Units.a0 * quad(lambda x: 1 / a(x), tmin, t)[0]

def particle_horizon_conformal(t: float, tmin: float = 0) -> float:
    """Particle horizon comoving distance. (Glyr)"""
    return Units.c_Glyr_per_Gyr * (t - tmin)

def event_horizon_conformal(t: float, t_max: float) -> float:
    """Event horizon comoving distance. (Glyr)"""
    return Units.c_Glyr_per_Gyr * (t_max - t)

def hubble_sphere_conformal(a: callable, t: float) -> float:
    # DOES NOT WORK
    """Hubble sphere comoving distance. (Glyr)"""
    return Units.c_Glyr_per_Gyr * a(t) / Units.H0_per_Gyr / Units.a0 / E(z(a, t))

def light_cone_conformal(today_conformal: float, tem: float) -> float:
    """Light cone comoving distance. (Glyr)"""
    if (tem > today_conformal):
        return np.nan
    return Units.c_Glyr_per_Gyr * (today_conformal - tem)

# ---------------- CONVERTING REDSHIFT TO DISTANCE ----------------

def chi_from_z(z: float) -> float:
    """Comoving distance from redshift z to today (Glyr)."""
    return Units.c_Glyr_per_Gyr / Units.H0_per_Gyr / Units.a0 \
        * quad(lambda x: 1 / E(x), 0, z)[0]

def iso_chi(a: callable, t: np.ndarray, chi: float) -> np.ndarray:
    """Isochrone redshift. (Glyr)"""
    return a(t) * chi

# ---------------------- COMPILED FUNCTIONS ----------------------

def horizons(a: callable, time: np.ndarray, large_time: np.ndarray) -> tuple:
    """Return all horizons."""
    # Particle Horizon
    p_h = np.zeros(len(time))
    for i in range(len(time)):
        p_h[i] = particule_horizon(a, time[i], tmin=large_time[0])

    # Event Horizon
    e_h = np.zeros(len(time))
    for i in range(len(time)):
        e_h[i] = event_horizon(a, time[i], t_max=large_time[-1])
    
    # Hubble Sphere
    h_s = np.zeros(len(time))
    for i in range(len(time)):
        h_s[i] = hubble_sphere(a, time[i])
    
    # Light Cone
    l_c = np.zeros(len(time))
    for i in range(len(time)):
        l_c[i] = light_cone(a, time[i])

    return p_h, e_h, h_s, l_c

def horizons_conformal(a: callable, time_conform: np.ndarray,
                       large_time: np.ndarray) -> tuple:
    """Return all horizons in conformal time."""

    p_h_comform = np.zeros(len(time_conform))
    for i in range(len(time_conform)):
        p_h_comform[i] = particle_horizon_conformal(time_conform[i], tmin=0)

    e_h_comform = np.zeros(len(time_conform))
    for i in range(len(time_conform)):
        e_h_comform[i] = event_horizon_conformal(time_conform[i],
                                                 t_max=time_conform[-1])

    h_s_comform = np.zeros(len(large_time))
    for i in range(len(large_time)):
        h_s_comform[i] =  hubble_sphere(a, large_time[i])

    l_c_comform = np.zeros(len(time_conform))
    for i in range(len(time_conform)):
        l_c_comform[i] = light_cone_conformal(today_conformal, time_conform[i])

    return p_h_comform, e_h_comform, h_s_comform, l_c_comform
    

# ------------------------- MAIN -------------------------

RECALC_PROPER_AND_COM = True
RECALC_CONFORMAL = True
SAVE = True

# NOTE : if you change the time scale, you need to recalculate everything,
# otherwise time scales won't match for plots, raising an error.

# COMMON PART : Time and Scale Factor
large_time = np.logspace(-10, 10, 10000)
sol = a_solve(large_time, ainit=Units.ainit, log=True)  # Scale factor
a_vals : np.ndarray = sol.y[0]
large_time = sol.t  # Time in Gyr
a = interp1d(large_time, a_vals)  # Interpolated scale factor
today_conformal = conformal_time(a, Units.today, tmin=large_time[0])

if RECALC_CONFORMAL:
    time_conform = np.zeros(len(large_time)) # Conformal Time in Gyr
    for i in range(len(large_time)):
        time_conform[i] = conformal_time(a, large_time[i], large_time[0])

    p_h, e_h, h_s, l_c = horizons_conformal(a, time_conform, large_time)

    data_conform = DataHolder()
    data_conform.set_all(a_vals, large_time, time_conform,
                        p_h, e_h, h_s, l_c)
    
    if SAVE:
        data_conform.save_data(folder="saved/conformal")

else:
    data_conform = DataHolder()
    data_conform.load_data(folder="saved/conformal")
    a = interp1d(data_conform.large_time, data_conform.a_vals)

if RECALC_PROPER_AND_COM:
    # Data intervals
    time = np.linspace(large_time[0], 25, 1000)  # Time in Gyr

    p_h, e_h, h_s, l_c = horizons(a, time, large_time)
    
    data = DataHolder()
    data_com = DataHolder()
    data.set_all(a_vals, large_time, time, p_h, e_h, h_s, l_c)
    data_com.copy_from(data)  # Copy data to data_com before converting to proper distance
    Units.convert_to_proper(a, data)  # Convert to proper distance

    if SAVE:
        data.save_data(folder="saved/proper")
        data_com.save_data(folder="saved/comoving")
else:
    # Load previously saved arrays
    data = DataHolder()
    data.load_data(folder="saved/proper")
    data_com = DataHolder()
    data_com.load_data(folder="saved/comoving")
    a = interp1d(data.large_time, data.a_vals)

# Calculating worldlines (or isochrones) for different redshifts for the
# proper distance plot.
time = np.linspace(large_time[0], 25, 1000)
zs = [1, 3, 10, 1000]
chis = [chi_from_z(z) for z in zs]
worldlines = []
for chi in chis:
    worldlines.append(iso_chi(a, time, chi))

Plotter.plot_horizons(data, a, name="horizons_proper",  worldlines=worldlines,
                      x_label="Proper Distance (Glyr)")
Plotter.plot_horizons(data_com, a, name="horizons_comoving",
                      x_label="Comoving Distance (Glyr)")
Plotter.plot_horizons(data_conform, a, name="horizons_conformal",
                      today=today_conformal,
                      x_label="Comoving Distance (Glyr)",
                      y_label="Conformal Time (Gyr)")
Plotter.plot_scale_factor(data, a_analytic)


FilteredOutput.off()
