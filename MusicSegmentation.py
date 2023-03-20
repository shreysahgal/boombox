import os
import pathlib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import ruptures as rpt

SAMPLE_RATE = 16000

class MusicSegmentation:
    bkps = -1
    
    def fig_ax(self, figsize=(15, 5), dpi=150):
        """Return a (matplotlib) figure and ax objects with given size."""
        return plt.subplots(figsize=figsize, dpi=dpi)
    
    def select_bkps(self, trajectory, n_bkps = 7):
        # Choose detection method
        algo = rpt.KernelCPD(kernel="linear").fit(trajectory)

        # Choose the number of changes (elbow heuristic)
        n_bkps_max = 20  # K_max
        # Start by computing the segmentation with most changes.
        # After start, all segmentations with 1, 2,..., K_max-1 changes are also available for free.
        _ = algo.predict(n_bkps_max)

        array_of_n_bkps = np.arange(1, n_bkps_max + 1)

        def get_sum_of_cost(algo, n_bkps) -> float:
            """Return the sum of costs for the change points `bkps`"""
            bkps = algo.predict(n_bkps=n_bkps)
            return algo.cost.sum_of_costs(bkps)

        fig, ax = self.fig_ax((7, 4))
        ax.plot(
            array_of_n_bkps,
            [get_sum_of_cost(algo=algo, n_bkps=n_bkps) for n_bkps in array_of_n_bkps],
            "-*",
            alpha=0.5,
        )
        ax.set_xticks(array_of_n_bkps)
        ax.set_xlabel("Number of change points")
        ax.set_title("Sum of costs")
        ax.grid(axis="x")
        ax.set_xlim(0, n_bkps_max + 1)

        # Visually we choose n_bkps=5 (highlighted in red on the elbow plot) 
        self.n_bkps = n_bkps                     
        _ = ax.scatter([n_bkps], [get_sum_of_cost(algo=algo, n_bkps=n_bkps)], color="r", s=100)
        
    def get_change_points(self, trajectory):
        algo = rpt.KernelCPD(kernel="linear").fit(trajectory)
        bkps = algo.predict(n_bkps=self.n_bkps)
        return bkps
