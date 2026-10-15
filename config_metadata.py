"""
config_metadata.py

Configuration and metadata models for the L4534A DAQ GUI.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any
import json


# ============================================================
# Instrument constants
# ============================================================

SAMPLE_RATE = 20_000_000  # 20 MSa/s

MIN_SAMPLES = 256
MAX_SAMPLES = 32_000_000

MIN_DURATION_MS = 0.01
MAX_DURATION_MS = 1600.0

SUPPORTED_RANGES = (
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
    128.0,
    256.0,
)

SUPPORTED_COUPLING = (
    "DC",
    "AC",
)

SUPPORTED_FILTERS = (
    "LP_20_MHZ",
    "LP_2_MHZ",
    "LP_200_KHZ",
)


# ============================================================
# Channel configuration
# ============================================================

@dataclass
class ChannelConfig:
    number: int
    name: str
    range_v: float
    coupling: str = "DC"
    filter_name: str = "LP_20_MHZ"
    enabled: bool = True

    def validate(self) -> None:
        if self.number not in (1, 2, 3, 4):
            raise ValueError(f"Invalid channel number: {self.number}")

        if self.range_v not in SUPPORTED_RANGES:
            raise ValueError(
                f"Channel {self.number}: invalid range {self.range_v} V"
            )

        if self.coupling not in SUPPORTED_COUPLING:
            raise ValueError(
                f"Channel {self.number}: invalid coupling {self.coupling}"
            )

        if self.filter_name not in SUPPORTED_FILTERS:
            raise ValueError(
                f"Channel {self.number}: invalid filter {self.filter_name}"
            )

        # dB/dt pickup should remain DC coupled.
        if self.number == 1 and self.coupling != "DC":
            raise ValueError(
                "Channel 1 (dB/dt Pickup) must use DC coupling."
            )


# ============================================================
# Timing
# ============================================================

@dataclass
class TimingConfig:
    duration_ms: float = 50.0
    pre_trigger_pct: float = 20.0

    @property
    def total_samples(self) -> int:
        samples = round(
            (self.duration_ms / 1000.0) * SAMPLE_RATE
        )

        return max(
            MIN_SAMPLES,
            min(MAX_SAMPLES, samples)
        )

    @property
    def pre_trigger_samples(self) -> int:
        value = round(
            self.total_samples *
            (self.pre_trigger_pct / 100.0)
        )

        return max(
            0,
            min(self.total_samples - 4, value)
        )

    @property
    def post_trigger_samples(self) -> int:
        return self.total_samples - self.pre_trigger_samples

    @property
    def actual_duration_ms(self) -> float:
        return (
            self.total_samples /
            SAMPLE_RATE *
            1000.0
        )

    @property
    def start_time_ms(self) -> float:
        return (
            -self.pre_trigger_samples /
            SAMPLE_RATE *
            1000.0
        )

    @property
    def end_time_ms(self) -> float:
        return (
            self.post_trigger_samples /
            SAMPLE_RATE *
            1000.0
        )

    def validate(self) -> None:
        if not (
            MIN_DURATION_MS <=
            self.duration_ms <=
            MAX_DURATION_MS
        ):
            raise ValueError(
                f"Duration must be between "
                f"{MIN_DURATION_MS} and "
                f"{MAX_DURATION_MS} ms."
            )

        if not (
            0.0 <= self.pre_trigger_pct <= 99.9
        ):
            raise ValueError(
                "Pre-trigger percentage must be between 0 and 99.9."
            )


# ============================================================
# Trigger
# ============================================================

@dataclass
class TriggerConfig:
    source: str = "EXT"
    slope: str = "POS"
    timeout_s: float = 180.0

    def validate(self, magnet_mode: bool = True) -> None:
        self.source = self.source.upper()
        self.slope = self.slope.upper()

        if self.source not in ("EXT", "IMM"):
            raise ValueError(
                "Trigger source must be EXT or IMM."
            )

        if self.slope not in ("POS", "NEG"):
            raise ValueError(
                "Trigger slope must be POS or NEG."
            )

        if self.timeout_s <= 0:
            raise ValueError(
                "Trigger timeout must be greater than zero."
            )

        if magnet_mode and self.source != "EXT":
            raise ValueError(
                "MAGNET SHOT mode requires EXT trigger."
            )


# ============================================================
# Shot metadata
# ============================================================

@dataclass
class ShotMetadata:
    shot_number: int = 1
    operator: str = ""
    coil: str = ""
    target_field: str = ""
    bank_voltage: str = ""
    notes: str = ""

    trigger_description: str = (
        "External master experiment trigger"
    )

    created_at: str = field(
        default_factory=lambda:
        datetime.now().isoformat(timespec="seconds")
    )

    software: str = "L4534A DAQ GUI"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Complete shot configuration
# ============================================================

@dataclass
class ShotConfiguration:

    mode: str = "MAGNET SHOT"

    timing: TimingConfig = field(
        default_factory=TimingConfig
    )

    trigger: TriggerConfig = field(
        default_factory=TriggerConfig
    )

    channels: list = field(
        default_factory=lambda: [
            ChannelConfig(
                1,
                "dB/dt Pickup",
                256.0,
                "DC",
                "LP_20_MHZ",
            ),
            ChannelConfig(
                2,
                "Sample Vxx",
                2.0,
                "DC",
                "LP_20_MHZ",
            ),
            ChannelConfig(
                3,
                "Sample Vxy",
                2.0,
                "DC",
                "LP_20_MHZ",
            ),
            ChannelConfig(
                4,
                "Aux Monitor",
                16.0,
                "DC",
                "LP_20_MHZ",
            ),
        ]
    )

    metadata: ShotMetadata = field(
        default_factory=ShotMetadata
    )

    save_npz: bool = True
    save_hdf5: bool = True
    save_csv: bool = False

    def validate(self) -> None:

        self.mode = self.mode.upper()

        if self.mode not in (
            "MAGNET SHOT",
            "BENCH TEST",
        ):
            raise ValueError(
                "Mode must be MAGNET SHOT or BENCH TEST."
            )

        self.timing.validate()

        self.trigger.validate(
            magnet_mode=(self.mode == "MAGNET SHOT")
        )

        for channel in self.channels:
            channel.validate()

        if self.mode == "MAGNET SHOT":
            if self.trigger.source != "EXT":
                raise ValueError(
                    "Real magnet shots must use EXT trigger."
                )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "mode": self.mode,
            "timing": asdict(self.timing),
            "trigger": asdict(self.trigger),
            "channels": [
                asdict(channel)
                for channel in self.channels
            ],
            "metadata": self.metadata.to_dict(),
            "save_npz": self.save_npz,
            "save_hdf5": self.save_hdf5,
            "save_csv": self.save_csv,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            indent=2
        )