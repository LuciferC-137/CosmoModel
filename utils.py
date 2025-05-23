import os
import numpy as np
from logger import Logger


def notnull(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result is None:
            raise ValueError(f"The value of {func.__name__} of the"
                             " DataHolder is None. You must either "
                             "load or set the data first.")
        return result
    return wrapper


class DataHolder:
    """Class to easily gather data for loading, saving and code organization."""

    def __init__(self) -> None:
        self._a_vals = None
        self._large_time = None
        self._time = None
        self._p_h = None
        self._e_h = None
        self._h_s = None
        self._l_c = None
    
    def is_empty(self) -> bool:
        """Check if any data is empty."""
        return (self.a_vals is None or self.large_time is None or
                self.time is None or self.p_h is None or
                self.e_h is None or self.h_s is None or
                self.l_c is None)
    
    def set_all(self, a_vals: np.ndarray,
                large_time: np.ndarray,
                time: np.ndarray,
                p_h: np.ndarray,
                e_h: np.ndarray,
                h_s: np.ndarray,
                l_c: np.ndarray) -> None:
        """"
        Set all data at once.
        In the case where conformal time is used, large_time should remain
        the time scale and time should be the conformal time.
        """
        self._a_vals = a_vals
        self._large_time = large_time
        self._time = time
        self._p_h = p_h
        self._e_h = e_h
        self._h_s = h_s
        self._l_c = l_c
    
    def load_data(self, folder: str = "saved") -> None:
        """Load data from files."""
        self._a_vals = DataHolder.load_array(os.path.join(folder, 'a_vals.txt'))
        self._large_time = DataHolder.load_array(os.path.join(folder,
                                                              'large_time.txt'))
        self._time = DataHolder.load_array(os.path.join(folder, 'time.txt'))
        self._p_h = DataHolder.load_array(os.path.join(folder, 'p_h.txt'))
        self._e_h = DataHolder.load_array(os.path.join(folder, 'e_h.txt'))
        self._h_s = DataHolder.load_array(os.path.join(folder, 'h_s.txt'))
        self._l_c = DataHolder.load_array(os.path.join(folder, 'l_c.txt'))
        Logger().log(f"Data loaded successfully from: {folder}")
    
    def save_data(self, folder: str = "saved") -> None:
        """Save data to files."""
        DataHolder.save_array(self.a_vals, os.path.join(folder, 'a_vals.txt'))
        DataHolder.save_array(self.large_time, os.path.join(folder,
                                                            'large_time.txt'))
        DataHolder.save_array(self.time, os.path.join(folder, 'time.txt'))
        DataHolder.save_array(self.p_h, os.path.join(folder, 'p_h.txt'))
        DataHolder.save_array(self.e_h, os.path.join(folder, 'e_h.txt'))
        DataHolder.save_array(self.h_s, os.path.join(folder, 'h_s.txt'))
        DataHolder.save_array(self.l_c, os.path.join(folder, 'l_c.txt'))
        Logger().log(f"Data saved successfully at: {folder}")

    def copy_from(self, data: 'DataHolder'):
        self.set_all(data.a_vals, data.large_time, data.time,
                     data.p_h, data.e_h, data.h_s, data.l_c)

    @staticmethod
    def save_array(array: np.ndarray, filename: str = "default") -> None:
        """Save a numpy array to a file."""
        np.savetxt(filename, array, delimiter=',')

    @staticmethod
    def load_array(filename: str = "default") -> np.ndarray:
        """Load a numpy array from a file."""
        return np.loadtxt(filename, delimiter=',')
    
    @property
    @notnull
    def a_vals(self) -> np.ndarray:
        return self._a_vals
    
    @property
    @notnull
    def large_time(self) -> np.ndarray:
        return self._large_time
    
    @property
    @notnull
    def time(self) -> np.ndarray:
        return self._time
    
    @property
    @notnull
    def p_h(self) -> np.ndarray:
        return self._p_h
    
    @property
    @notnull
    def e_h(self) -> np.ndarray:
        return self._e_h
    
    @property
    @notnull
    def h_s(self) -> np.ndarray:
        return self._h_s
    
    @property
    @notnull
    def l_c(self) -> np.ndarray:
        return self._l_c


class Units:
    """Class to hold constants."""
    c = 299792.458  # Speed of light in km/s
    H0 = 67.15  # Hubble constant in km/s/Mpc
    Omega_m = 0.315  # Matter density parameter
    Omega_r = 0.000092136  # Radiation density parameter
    Omega_lambda = 1. - Omega_m - Omega_r # Dark energy density parameter

    today = 13.842  # Age of the universe in Gyr
    a0 = 1.0  # Scale factor today
    ainit = 1e-10  # Initial scale factor for integration.

    sec_in_yr = 365.25 * 86400
    sec_in_Gyr = sec_in_yr * 1e9
    km_to_Mpc = 1 / 3.085e19
    km_to_Gpc = km_to_Mpc / 1e3
    pc_to_ly = 3.261
    H0_per_Gyr = sec_in_Gyr * H0 * km_to_Mpc # Hubble constant in Gyr^-1
    c_Glyr_per_Gyr = c * sec_in_Gyr * km_to_Gpc * pc_to_ly
    Lambda = Omega_lambda * 3 * H0 * H0 * km_to_Mpc * km_to_Mpc

    @staticmethod
    def proper(a: callable, t: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Proper distance from scale factor, time and comoving distance."""
        return a(t) / Units.a0 * d
    
    @staticmethod
    def convert_to_proper(a: callable, data: DataHolder):
        """Convert comoving distance to proper distance inside a dataset."""
        data._p_h = Units.proper(a, data.time, data.p_h)
        data._e_h = Units.proper(a, data.time, data.e_h)
        data._h_s = Units.proper(a, data.time, data.h_s)
        data._l_c = Units.proper(a, data.time, data.l_c)

