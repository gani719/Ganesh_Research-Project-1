"""
plotter.py

Waveform plotting and basic shot diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt


class ShotPlotter:

    def __init__(
        self,
        max_display_points: int = 50_000
    ):

        self.max_display_points = (
            max_display_points
        )

    # ========================================================
    # Downsampling for display only
    # ========================================================

    def _display_indices(
        self,
        n: int
    ):

        if n <= self.max_display_points:

            return np.arange(n)

        step = int(
            np.ceil(
                n /
                self.max_display_points
            )
        )

        return np.arange(
            0,
            n,
            step
        )

    # ========================================================
    # Diagnostics
    # ========================================================

    @staticmethod
    def diagnostics(
        data,
        channels,
    ):

        result = {}

        for channel in channels:

            values = np.asarray(
                data[channel],
                dtype=float
            )

            cfg = channels[channel]

            peak = float(
                np.max(np.abs(values))
            )

            rms = float(
                np.sqrt(
                    np.mean(values ** 2)
                )
            )

            range_v = float(
                cfg.range_v
            )

            saturation = (
                peak >=
                0.98 * range_v
            )

            flatline = (
                np.ptp(values) <
                max(
                    range_v * 1e-6,
                    1e-12
                )
            )

            result[channel] = {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "peak_abs": peak,
                "rms": rms,
                "p2p": float(np.ptp(values)),
                "range_v": range_v,
                "saturation": saturation,
                "flatline": flatline,
            }

        return result

    # ========================================================
    # Plot
    # ========================================================

    def plot(
        self,
        time_ms,
        data,
        channel_configs,
        shot_number=None,
    ):

        indices = self._display_indices(
            len(time_ms)
        )

        fig, axes = plt.subplots(
            4,
            1,
            figsize=(12, 9),
            sharex=True,
        )

        if shot_number is not None:

            fig.suptitle(
                f"L4534A Shot {shot_number}",
                fontsize=14,
            )

        for axis, channel in zip(
            axes,
            range(1, 5)
        ):

            cfg = channel_configs[channel]

            axis.plot(
                time_ms[indices],
                data[channel][indices],
                linewidth=0.8,
            )

            axis.axvline(
                0.0,
                linestyle="--",
                linewidth=1.0,
            )

            axis.set_ylabel(
                f"CH{channel}\n"
                f"{cfg.name}"
            )

            axis.grid(
                True,
                alpha=0.25
            )

        axes[-1].set_xlabel(
            "Time relative to master trigger (ms)"
        )

        fig.tight_layout()

        return fig

    # ========================================================
    # Show
    # ========================================================

    def show(
        self,
        time_ms,
        data,
        channel_configs,
        shot_number=None,
    ):

        fig = self.plot(
            time_ms,
            data,
            channel_configs,
            shot_number
        )

        plt.show()

        return fig