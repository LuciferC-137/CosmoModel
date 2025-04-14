import numpy as np
from scipy.interpolate import interp1d
from utils import DataHolder, Units
from plotter import Plotter
from logger import FilteredOutput, Logger
from cosmo_methods import *

Logger().log("Starting...")

# Filtering warnings for a cleaner output. When this is on, 
# all print() functions must explicitly end with '\n' to make a new line.
FilteredOutput.on()


# ---------------------- COMPILED FUNCTIONS ----------------------

def horizons(a: callable, time: np.ndarray, large_time: np.ndarray) -> tuple:
    """Return all horizons."""
    # Particle Horizon
    p_h = particule_horizon(a(large_time), time, large_time)
    Logger().log_prc_done("Particle horizon")

    # Event Horizon
    e_h = event_horizon(a(large_time), time, large_time)
    Logger().log_prc_done("Event horizon")
    
    # Hubble Sphere
    h_s = hubble_sphere(a(time), time)
    Logger().log_prc_done("Hubble sphere")
    
    # Light Cone
    l_c = light_cone(a(time), time)
    Logger().log_prc_done("Light cone")

    return p_h, e_h, h_s, l_c


def horizons_conformal(a: callable, time_conform: np.ndarray,
                       large_time: np.ndarray) -> tuple:
    """Return all horizons in conformal time."""
    n = len(time_conform)
    p_h_comform = np.zeros(n)
    for i in range(n):
        Logger().log_prc("Particle horizon conformal", i, n)
        p_h_comform[i] = particle_horizon_conformal(time_conform[i], tmin=0)
    Logger().log_prc_done("Particle horizon conformal distance")

    e_h_comform = np.zeros(n)
    for i in range(n):
        Logger().log_prc("Event horizon conformal", i, n)
        e_h_comform[i] = event_horizon_conformal(time_conform[i],
                                                 t_max=time_conform[-1])
    Logger().log_prc_done("Event horizon conformal")

    h_s_comform =  hubble_sphere(a(large_time), large_time[i])
    Logger().log_prc_done("Hubble sphere conformal")

    l_c_comform = np.zeros(n)
    for i in range(n):
        Logger().log_prc("Light cone conformal", i, n)
        l_c_comform[i] = light_cone_conformal(today_conformal, time_conform[i])
    Logger().log_prc_done("Light cone conformal")

    return p_h_comform, e_h_comform, h_s_comform, l_c_comform
    

# ------------------------- MAIN -------------------------

RECALC_PROPER_AND_COM = True
RECALC_CONFORMAL = True
SAVE = True

# NOTE : if you change the time scale, you need to recalculate everything,
# otherwise time scales won't match for plots, raising an error.

# COMMON PART : Time and Scale Factor
large_time = np.logspace(-10, 10, 10000)
Logger().log("Calculating scale factor (handled by solve_ivp)...")
sol = a_solve(large_time, ainit=Units.ainit, log=True)  # Scale factor
a_vals : np.ndarray = sol.y[0]
large_time = sol.t  # Time in Gyr
a = interp1d(large_time, a_vals)  # Interpolated scale factor
a_vals_full = a(large_time)
today_conformal = conformal_time(a, Units.today, tmin=large_time[0])

if RECALC_CONFORMAL:
   
    time_conform = conformal_time_vect(a_vals_full, large_time)

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
    Logger().log("Beginning proper and comoving distance calculations...")
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
zs = [0, 1, 3, 10, 1000, 1e10]
chis = [chi_from_z(z) for z in zs]
worldlines = []
for chi in chis:
    Logger().log_prc("Calculating worldlines", chi, chis[-1])
    worldlines.append(iso_chi(a, time, chi))
Logger().log_prc_done("Calculating worldlines")

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
