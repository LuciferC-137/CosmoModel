import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from utils import DataHolder, Units


class Plotter:

    @staticmethod
    def sym_plot(ax: Axes, x: np.ndarray, y: np.ndarray,
                  label: str = '', color: str = 'blue',
                  style: str = "-") -> None:
        """Easy plot function to plot symetries."""
        ax.plot(x, y, style, label=label, color=color)
        ax.plot(-x, y, style, color=color)
    
    @staticmethod
    def plot_horizons(data: DataHolder, a: callable, today: float = Units.today,
                      worldlines: list[np.ndarray] = [],
                      name: str = "horizons",
                      x_label: str = "Distance (Glyr)",
                      y_label: str = "Time (Glyr)") -> None:
        fig, ax = plt.subplots(figsize=(12, 4))
        Plotter.sym_plot(ax, data.p_h, data.time,
                          label='Particle Horizon', color='blue')
        Plotter.sym_plot(ax, data.e_h, data.time,
                          label='Event Horizon', color='red')
        Plotter.sym_plot(ax, data.h_s, data.time,
                          label='Hubble Sphere', color='purple')
        Plotter.sym_plot(ax, data.l_c, data.time,
                          label='Light Cone', color='black')
        for d in worldlines:
            Plotter.sym_plot(ax, d, data.time, style = '--',
                              color='gray')
        ax.plot([-60, 60], [today, today], 'k-', linewidth=0.5)
        ax.grid()
        ax.legend()
        ax.set_xlim(-60, 60)
        ax.set_ylim(data.time[0], data.time[-1])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Plotting the scale factor on the right y-axis
        ax_right = ax.twinx()
        ax_right.set_ylim(ax.get_ylim())
        additional_ticks = np.arange(data.large_time[0], data.time[-1], 4)
        ax_right.set_yticks(additional_ticks,
                            [round(sc_fc, 2) for sc_fc in a(additional_ticks)])
        ax_right.tick_params(axis='y', direction = 'in')
        ax_right.set_ylabel('scale factor')
        if name[:-4] != ".png":
            name += ".png"
        plt.savefig(name, dpi=300, bbox_inches='tight')
        print(f"Figure saved as '{name}'\n")

    @staticmethod
    def plot_scale_factor(data: DataHolder, a_analytic: callable) -> None:
        a_analytic_vals = a_analytic(data.large_time)
        fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(12, 4))
        ax1 : Axes
        ax2 : Axes
        ax1.plot(data.large_time, data.a_vals,
                 label='Scale Factor', color='green')
        ax1.plot(data.large_time, a_analytic_vals, '--',
                 label='Analytic Scale Factor', color='orange')
        ax1.set_yscale('log') 
        ax1.set_xlabel('Time (Gyr)')
        ax1.set_ylabel('Scale Factor')
        ax1.set_title('Scale Factor ' + r"[0; 1000] Gyr" + ' (Log Scale)')
        mask = data.large_time < 50
        large_time = data.large_time[mask]
        a_vals = data.a_vals[mask]
        a_analytic_vals = a_analytic_vals[mask]
        ax2.plot(large_time, a_vals, label='Scale Factor', color='green')
        ax2.plot(large_time, a_analytic_vals, '--',
                 label='Analytic Scale Factor', color='orange') 
        ax2.set_xlabel('Time (Gyr)')
        ax2.set_ylabel('Scale Factor')
        ax2.set_title('Scale Factor '+ r"[0; 50] Gyr" + '(Linear Scale)')
        plt.savefig('scale_factor.png', dpi=300, bbox_inches='tight')
        print(f"Figure saved as 'scale_factor.png'\n")
