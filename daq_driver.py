"""
daq_driver.py

Keysight / Agilent L4534A driver.

Real experiment workflow:

    connect
        ↓
    configure
        ↓
    INIT
        ↓
    WAIT FOR EXTERNAL TRIGGER
        ↓
    master magnet trigger
        ↓
    acquisition complete
        ↓
    FETCH waveform
"""

import time
import threading
from typing import Callable, Optional, Dict

import numpy as np
import pyvisa


DEFAULT_VISA_ADDRESS = (
    "USB0::0x0957::0x1307::MY48000262::0::INSTR"
)

DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024


class L4534AError(RuntimeError):
    """Driver-level error."""


class L4534ADriver:

    def __init__(
        self,
        visa_address: str = DEFAULT_VISA_ADDRESS,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        reset_instrument: bool = False,
    ):

        self.visa_address = visa_address
        self.chunk_size = chunk_size
        self.reset_instrument = reset_instrument

        self.rm = None
        self.inst = None

        self.connected = False

        self._abort_event = threading.Event()

    # ========================================================
    # Connection
    # ========================================================

    def connect(self):

        if self.connected:
            return

        self.rm = pyvisa.ResourceManager()

        self.inst = self.rm.open_resource(
            self.visa_address
        )

        self.inst.chunk_size = self.chunk_size

        self.inst.read_termination = "\n"
        self.inst.write_termination = "\n"

        # Connection timeout.
        self.inst.timeout = 5000

        identity = self.inst.query("*IDN?").strip()

        self.connected = True

        if self.reset_instrument:
            self.reset()

        return identity

    def close(self):

        try:
            if self.inst is not None:
                self.inst.close()
        finally:
            self.inst = None
            self.connected = False

            if self.rm is not None:
                self.rm.close()

            self.rm = None

    # ========================================================
    # Basic SCPI helpers
    # ========================================================

    def write(self, command: str):
        if not self.connected:
            raise L4534AError(
                "Digitizer is not connected."
            )

        self.inst.write(command)

    def query(self, command: str):
        if not self.connected:
            raise L4534AError(
                "Digitizer is not connected."
            )

        return self.inst.query(command).strip()

    # ========================================================
    # Reset / abort
    # ========================================================

    def reset(self):

        self.write("*RST")
        time.sleep(1.0)
        self.write("*CLS")

    def abort(self):

        self._abort_event.set()

        if self.connected:
            try:
                self.write("ABOR")
            except Exception:
                pass

    # ========================================================
    # Identification
    # ========================================================

    def identify(self) -> str:
        return self.query("*IDN?")

    # ========================================================
    # Configuration
    # ========================================================

    def configure(self, shot_config):

        shot_config.validate()

        timing = shot_config.timing
        trigger = shot_config.trigger

        # Clear old status/events.
        self.write("*CLS")

        # ----------------------------------------------------
        # Acquisition
        #
        # 20 MSa/s
        # total samples
        # pre-trigger samples
        # 1 record
        # zero holdoff
        # zero trigger delay
        # ----------------------------------------------------

        self.write(
            "CONF:ACQ "
            f"{20_000_000},"
            f"{timing.total_samples},"
            f"{timing.pre_trigger_samples},"
            "1,0,0"
        )

        # ----------------------------------------------------
        # Channels
        # ----------------------------------------------------

        for channel in shot_config.channels:

            if not channel.enabled:
                continue

            self.write(
                "CONF:CHAN:ATTR "
                f"(@{channel.number}),"
                f"{channel.range_v},"
                f"{channel.coupling},"
                f"{channel.filter_name}"
            )

        # ----------------------------------------------------
        # ARM source
        #
        # IMPORTANT:
        # Real magnet shot:
        #
        #     ARM = IMM
        #
        # The digitizer becomes armed immediately after INIT.
        #
        # We do NOT use external arm because the same master
        # pulse must be the trigger that defines t = 0.
        # ----------------------------------------------------

        self.write("CONF:ARM:SOUR IMM,1")

        # ----------------------------------------------------
        # Trigger source
        # ----------------------------------------------------

        if trigger.source == "EXT":

            self.write("CONF:TRIG:SOUR EXT")
            self.write(
                f"CONF:EXT:INP {trigger.slope}"
            )

        elif trigger.source == "IMM":

            self.write("CONF:TRIG:SOUR IMM")

        else:
            raise L4534AError(
                f"Unsupported trigger source: "
                f"{trigger.source}"
            )

        # ----------------------------------------------------
        # No trigger output from the digitizer.
        #
        # The experiment's master trigger is an input.
        # ----------------------------------------------------

        self.write(
            "CONF:EXT:TRIG:OUTP NONE,OFF"
        )

        # ----------------------------------------------------
        # Trigger timestamp mode.
        #
        # INIT mode means the timestamp reference is reset
        # at every acquisition.
        # ----------------------------------------------------

        self.write(
            "CONF:TRIG:TIME INIT"
        )

        # ----------------------------------------------------
        # Binary waveform output
        #
        # L4534A supports REAL32 and REAL64 for
        # FETC:WAV:VOLT?
        # ----------------------------------------------------

        self.write(
            "FORM:DATA REAL,64"
        )

        # L4534A defaults to big-endian.
        self.write(
            "FORM:BORD NORM"
        )

        # Make timeout large enough for transfer.
        self.inst.timeout = self._calculate_timeout(
            timing.total_samples
        )

    # ========================================================
    # Configuration verification
    # ========================================================

    def verify_configuration(self, shot_config):

        expected_trigger = shot_config.trigger.source

        actual_trigger = self.query(
            "CONF:TRIG:SOUR?"
        ).upper()

        if expected_trigger == "EXT":
            if "EXT" not in actual_trigger:
                raise L4534AError(
                    "Digitizer did not accept external trigger "
                    f"configuration. Returned: {actual_trigger}"
                )

        # Query sample rate.
        try:
            actual_rate = float(
                self.query("CONF:ACQ:SRAT?")
            )

            if abs(actual_rate - 20_000_000) > 1:
                raise L4534AError(
                    f"Unexpected sample rate: {actual_rate}"
                )

        except ValueError:
            # Some firmware versions may format the response
            # differently. Do not fail only because of parsing.
            pass

    # ========================================================
    # Timeout calculation
    # ========================================================

    @staticmethod
    def _calculate_timeout(
        total_samples: int
    ) -> int:

        # Acquisition itself may be up to 1.6 s.
        #
        # Transfer over USB is much slower than acquisition,
        # so give it generous headroom.
        transfer_seconds = (
            total_samples * 8 * 4
        ) / 8_000_000

        return int(
            max(
                30,
                min(
                    600,
                    20 + transfer_seconds
                )
            ) * 1000
        )

    # ========================================================
    # Start acquisition / ARM
    # ========================================================

    def arm(
        self,
        shot_config,
        status_callback: Optional[
            Callable[[str], None]
        ] = None,
    ):

        if not self.connected:
            raise L4534AError(
                "Digitizer is not connected."
            )

        shot_config.validate()

        self._abort_event.clear()

        self.configure(shot_config)
        self.verify_configuration(shot_config)

        if status_callback:
            status_callback(
                "INITIALIZING DIGITIZER"
            )

        # Standard event status setup.
        #
        # *OPC sets Standard Event Status bit 0
        # when the INIT acquisition is complete.
        #
        # *ESE 1 routes OPC to Event Status.
        # *SRE 32 routes Event Status to Status Byte.
        #
        self.write("*CLS")
        self.write("*ESE 1")
        self.write("*SRE 32")

        # INIT moves the digitizer out of IDLE and into
        # its arm/trigger state.
        self.write("INIT")

        # Immediately issue OPC WITHOUT querying it.
        # This does not block.
        self.write("*OPC")

        if status_callback:

            if shot_config.trigger.source == "EXT":
                status_callback(
                    "ARMED — WAITING FOR MAGNET TRIGGER"
                )
            else:
                status_callback(
                    "ARMED — IMMEDIATE TRIGGER"
                )

    # ========================================================
    # Wait for acquisition completion
    # ========================================================

    def wait_for_completion(
        self,
        timeout_s: float,
        status_callback: Optional[
            Callable[[str], None]
        ] = None,
    ):

        start = time.monotonic()
        last_report = 0.0

        while True:

            if self._abort_event.is_set():
                raise L4534AError(
                    "Acquisition aborted by user."
                )

            elapsed = (
                time.monotonic() - start
            )

            if elapsed > timeout_s:

                self.abort()

                raise L4534AError(
                    "Trigger/acquisition timeout "
                    f"after {timeout_s:.1f} s."
                )

            # Read status byte without sending a SCPI query.
            stb = self.inst.read_stb()

            # ESB = bit 5.
            if stb & 32:

                # Clear/read the event status register.
                esr = int(
                    self.query("*ESR?")
                )

                # OPC = bit 0.
                if esr & 1:
                    break

            # Status update every ~0.5 s.
            if (
                status_callback is not None
                and elapsed - last_report > 0.5
            ):

                status_callback(
                    "WAITING / ACQUIRING "
                    f"({elapsed:.1f} s)"
                )

                last_report = elapsed

            time.sleep(0.05)

    # ========================================================
    # Fetch waveforms
    # ========================================================

    def fetch_all_channels(
        self,
        expected_samples: int,
        progress_callback: Optional[
            Callable[[int, int], None]
        ] = None,
    ) -> Dict[int, np.ndarray]:

        if not self.connected:
            raise L4534AError(
                "Digitizer is not connected."
            )

        data = {}

        for index, channel in enumerate(
            (1, 2, 3, 4),
            start=1,
        ):

            if self._abort_event.is_set():
                raise L4534AError(
                    "Data transfer aborted."
                )

            # Fetch voltage data directly.
            #
            # L4534A documented form:
            #
            # FETC:WAV:VOLT? (@1)
            #
            command = (
                f"FETC:WAV:VOLT? (@{channel})"
            )

            values = self.inst.query_binary_values(
                command,
                datatype="d",
                is_big_endian=True,
                container=np.array,
            )

            values = np.asarray(
                values,
                dtype=np.float64
            )

            # Check returned length.
            if expected_samples > 0:

                if len(values) != expected_samples:

                    raise L4534AError(
                        f"CH{channel}: expected "
                        f"{expected_samples:,} samples, "
                        f"received {len(values):,}."
                    )

            data[channel] = values

            if progress_callback:
                progress_callback(
                    index,
                    4
                )

        return data

    # ========================================================
    # Complete acquisition helper
    # ========================================================

    def acquire(
        self,
        shot_config,
        status_callback: Optional[
            Callable[[str], None]
        ] = None,
        progress_callback: Optional[
            Callable[[int, int], None]
        ] = None,
    ):

        self.arm(
            shot_config,
            status_callback=status_callback
        )

        self.wait_for_completion(
            shot_config.trigger.timeout_s,
            status_callback=status_callback
        )

        if status_callback:
            status_callback(
                "ACQUISITION COMPLETE — TRANSFERRING DATA"
            )

        data = self.fetch_all_channels(
            shot_config.timing.total_samples,
            progress_callback=progress_callback,
        )

        if status_callback:
            status_callback(
                "DATA TRANSFER COMPLETE"
            )

        return data

    # ========================================================
    # Instrument error queue
    # ========================================================

    def read_errors(self):

        errors = []

        for _ in range(20):

            try:
                response = self.query(
                    "SYST:ERR?"
                )
            except Exception:
                break

            errors.append(response)

            if response.startswith("0"):
                break

        return errors