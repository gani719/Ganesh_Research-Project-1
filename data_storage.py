"""
data_storage.py

Shot data storage for the L4534A DAQ GUI.
"""

from pathlib import Path
from datetime import datetime
import json
import csv

import numpy as np


DATA_ROOT = Path("data_runs")


class ShotStorage:

    def __init__(
        self,
        root: Path = DATA_ROOT
    ):
        self.root = Path(root)

    # ========================================================
    # Shot directory
    # ========================================================

    def create_shot_directory(
        self,
        shot_number: int
    ) -> Path:

        date_folder = datetime.now().strftime(
            "%Y%m%d"
        )

        shot_folder = (
            self.root /
            date_folder /
            f"Shot_{shot_number:03d}"
        )

        shot_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        return shot_folder

    # ========================================================
    # Time axis
    # ========================================================

    @staticmethod
    def make_time_axis(
        total_samples: int,
        pre_trigger_samples: int,
        sample_rate: float = 20_000_000,
    ):

        indices = np.arange(
            total_samples,
            dtype=np.float64
        )

        return (
            (indices - pre_trigger_samples)
            / sample_rate
            * 1000.0
        )

    # ========================================================
    # Save NPZ
    # ========================================================

    def save_npz(
        self,
        folder: Path,
        time_ms,
        data,
        metadata,
    ):

        path = folder / "shot.npz"

        np.savez_compressed(
            path,
            time_ms=time_ms,
            ch1=data[1],
            ch2=data[2],
            ch3=data[3],
            ch4=data[4],
        )

        return path

    # ========================================================
    # Save HDF5
    # ========================================================

    def save_hdf5(
        self,
        folder: Path,
        time_ms,
        data,
        metadata,
    ):

        path = folder / "shot.h5"

        try:
            import h5py
        except ImportError:
            raise RuntimeError(
                "h5py is not installed. "
                "Install it with: pip install h5py"
            )

        with h5py.File(path, "w") as h5:

            h5.create_dataset(
                "time_ms",
                data=time_ms,
                compression="gzip"
            )

            for channel in range(1, 5):

                h5.create_dataset(
                    f"ch{channel}",
                    data=data[channel],
                    compression="gzip"
                )

            metadata_group = h5.create_group(
                "metadata"
            )

            for key, value in metadata.items():

                metadata_group.attrs[
                    key
                ] = str(value)

        return path

    # ========================================================
    # Save CSV
    # ========================================================

    def save_csv(
        self,
        folder: Path,
        time_ms,
        data,
        max_points: int = 100_000,
    ):

        path = folder / "shot.csv"

        n = len(time_ms)

        if n > max_points:

            step = int(
                np.ceil(
                    n / max_points
                )
            )

            indices = np.arange(
                0,
                n,
                step
            )

        else:

            indices = np.arange(n)

        with path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "time_ms",
                "ch1_dBdt",
                "ch2_Vxx",
                "ch3_Vxy",
                "ch4_aux",
            ])

            for i in indices:

                writer.writerow([
                    time_ms[i],
                    data[1][i],
                    data[2][i],
                    data[3][i],
                    data[4][i],
                ])

        return path

    # ========================================================
    # Metadata
    # ========================================================

    def save_metadata(
        self,
        folder: Path,
        metadata: dict,
    ):

        path = folder / "metadata.json"

        with path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=2,
                default=str
            )

        return path

    # ========================================================
    # Complete shot save
    # ========================================================

    def save_shot(
        self,
        shot_config,
        data,
    ):

        folder = self.create_shot_directory(
            shot_config.metadata.shot_number
        )

        time_ms = self.make_time_axis(
            shot_config.timing.total_samples,
            shot_config.timing.pre_trigger_samples,
        )

        metadata = shot_config.to_dict()

        metadata["storage"] = {
            "folder": str(folder),
            "sample_rate_hz": 20_000_000,
            "total_samples": shot_config.timing.total_samples,
            "pre_trigger_samples":
                shot_config.timing.pre_trigger_samples,
            "post_trigger_samples":
                shot_config.timing.post_trigger_samples,
            "time_start_ms":
                shot_config.timing.start_time_ms,
            "time_end_ms":
                shot_config.timing.end_time_ms,
        }

        saved_files = []

        if shot_config.save_npz:

            saved_files.append(
                str(
                    self.save_npz(
                        folder,
                        time_ms,
                        data,
                        metadata,
                    )
                )
            )

        if shot_config.save_hdf5:

            try:

                saved_files.append(
                    str(
                        self.save_hdf5(
                            folder,
                            time_ms,
                            data,
                            metadata,
                        )
                    )
                )

            except ImportError:
                # HDF5 is optional.
                pass

        if shot_config.save_csv:

            saved_files.append(
                str(
                    self.save_csv(
                        folder,
                        time_ms,
                        data,
                    )
                )
            )

        metadata["saved_files"] = saved_files

        self.save_metadata(
            folder,
            metadata
        )

        return {
            "folder": folder,
            "time_ms": time_ms,
            "files": saved_files,
            "metadata": metadata,
        }