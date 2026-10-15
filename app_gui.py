"""
app_gui.py

Simple operator GUI for the Keysight/Agilent L4534A.

Real experiment:

    Configure
        ↓
    ARM DIGITIZER
        ↓
    WAITING FOR MAGNET TRIGGER
        ↓
    external master pulse
        ↓
    acquisition
        ↓
    transfer
        ↓
    save

Python does NOT trigger the capacitor bank.
"""

import json
import queue
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from config_metadata import (
    ShotConfiguration,
    ChannelConfig,
    TimingConfig,
    TriggerConfig,
    ShotMetadata,
    SUPPORTED_RANGES,
    SUPPORTED_COUPLING,
    SUPPORTED_FILTERS,
)

from daq_driver import (
    L4534ADriver,
    L4534AError,
    DEFAULT_VISA_ADDRESS,
)

from data_storage import ShotStorage
from plotter import ShotPlotter


class L4534AGUI:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "L4534A Pulsed-Magnet DAQ"
        )

        # Fit the complete control interface inside the available screen.
        # The previous fixed 1250x820 geometry could extend below shorter
        # displays, making the lower controls appear to require scrolling.
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        window_w = min(1450, max(1000, screen_w - 60))
        window_h = min(900, max(620, screen_h - 100))
        self.root.geometry(
            f"{window_w}x{window_h}+{max(0, (screen_w-window_w)//2)}+"
            f"{max(0, (screen_h-window_h)//2)}"
        )
        self.root.minsize(960, 600)

        self.driver = None

        self.storage = ShotStorage()
        self.plotter = ShotPlotter()

        self.events = queue.Queue()

        self.worker = None

        self.abort_requested = False

        self.last_data = None
        self.last_time = None

        self.current_shot_folder = None

        self._build_variables()
        self._build_gui()

        self.root.after(
            100,
            self._process_events
        )

        self._update_preview()

    # ========================================================
    # Variables
    # ========================================================

    def _build_variables(self):

        self.mode_var = tk.StringVar(
            value="MAGNET SHOT"
        )

        self.duration_var = tk.StringVar(
            value="50"
        )

        self.pretrigger_var = tk.StringVar(
            value="20"
        )

        self.trigger_var = tk.StringVar(
            value="EXT"
        )

        self.slope_var = tk.StringVar(
            value="POS"
        )

        self.timeout_var = tk.StringVar(
            value="180"
        )

        self.shot_number_var = tk.StringVar(
            value="1"
        )

        self.operator_var = tk.StringVar()

        self.coil_var = tk.StringVar()

        self.target_field_var = tk.StringVar()

        self.bank_voltage_var = tk.StringVar()

        self.notes_var = tk.StringVar()

        self.preview_var = tk.StringVar()

        self.status_var = tk.StringVar(
            value="DISCONNECTED"
        )

        self.progress_var = tk.StringVar(
            value="0 / 4"
        )

        self.visa_var = tk.StringVar(
            value=DEFAULT_VISA_ADDRESS
        )

        self.chunk_var = tk.StringVar(
            value=str(
                4 * 1024 * 1024
            )
        )

        self.reset_var = tk.BooleanVar(
            value=False
        )

        self.npz_var = tk.BooleanVar(
            value=True
        )

        self.hdf5_var = tk.BooleanVar(
            value=True
        )

        self.csv_var = tk.BooleanVar(
            value=False
        )

        # Channel variables.
        self.channel_vars = {}

        defaults = [
            (
                "dB/dt Pickup",
                256.0,
                "DC",
                "LP_20_MHZ",
            ),
            (
                "Sample Vxx",
                2.0,
                "DC",
                "LP_20_MHZ",
            ),
            (
                "Sample Vxy",
                2.0,
                "DC",
                "LP_20_MHZ",
            ),
            (
                "Aux Monitor",
                16.0,
                "DC",
                "LP_20_MHZ",
            ),
        ]

        for ch, values in enumerate(
            defaults,
            start=1
        ):

            name, rng, coupling, filt = values

            self.channel_vars[ch] = {
                "name": tk.StringVar(
                    value=name
                ),
                "range": tk.StringVar(
                    value=str(rng)
                ),
                "coupling": tk.StringVar(
                    value=coupling
                ),
                "filter": tk.StringVar(
                    value=filt
                ),
                "enabled": tk.BooleanVar(
                    value=True
                ),
            }

    # ========================================================
    # GUI
    # ========================================================

    def _build_gui(self):

        root = self.root

        root.columnconfigure(
            0,
            weight=1
        )

        root.rowconfigure(
            4,
            weight=1
        )

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        header = ttk.Frame(
            root,
            padding=(10, 6)
        )

        header.grid(
            row=0,
            column=0,
            sticky="ew"
        )

        header.columnconfigure(
            1,
            weight=1
        )

        ttk.Label(
            header,
            text="L4534A Pulsed-Magnet DAQ",
            font=("Segoe UI", 18, "bold"),
        ).grid(
            row=0,
            column=0,
            sticky="w"
        )

        status_frame = ttk.Frame(
            header
        )

        status_frame.grid(
            row=0,
            column=1,
            sticky="e"
        )

        ttk.Label(
            status_frame,
            text="STATUS:"
        ).pack(
            side="left",
            padx=(0, 6)
        )

        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 11, "bold"),
        ).pack(
            side="left"
        )

        # ----------------------------------------------------
        # Timing / experiment
        # ----------------------------------------------------

        timing = ttk.LabelFrame(
            root,
            text="Shot Timing",
            padding=7
        )

        timing.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=15,
            pady=(0, 5)
        )

        for col in range(12):
            timing.columnconfigure(
                col,
                weight=1
            )

        self._add_labeled_entry(
            timing,
            "Mode",
            self.mode_var,
            0,
            0,
            combo=True,
            values=[
                "MAGNET SHOT",
                "BENCH TEST",
            ],
            callback=self._mode_changed,
        )

        self._add_labeled_entry(
            timing,
            "Duration (ms)",
            self.duration_var,
            0,
            2,
        )

        self._add_labeled_entry(
            timing,
            "Pre-trigger (%)",
            self.pretrigger_var,
            0,
            4,
        )

        self._add_labeled_entry(
            timing,
            "Trigger",
            self.trigger_var,
            0,
            6,
            combo=True,
            values=["EXT", "IMM"],
            callback=self._trigger_changed,
        )

        self._add_labeled_entry(
            timing,
            "Slope",
            self.slope_var,
            0,
            8,
            combo=True,
            values=["POS", "NEG"],
        )

        self._add_labeled_entry(
            timing,
            "Trigger Timeout (s)",
            self.timeout_var,
            0,
            10,
            width=8,
        )

        ttk.Label(
            timing,
            textvariable=self.preview_var,
            font=("Segoe UI", 10)
        ).grid(
            row=1,
            column=0,
            columnspan=12,
            sticky="w",
            pady=(10, 0)
        )

        # ----------------------------------------------------
        # Channels
        # ----------------------------------------------------

        channels = ttk.LabelFrame(
            root,
            text="Channels",
            padding=6
        )

        channels.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=15,
            pady=(0, 5)
        )

        headings = [
            "Use",
            "Channel",
            "Name",
            "Range",
            "Coupling",
            "Filter",
        ]

        for col, heading in enumerate(
            headings
        ):

            ttk.Label(
                channels,
                text=heading,
                font=("Segoe UI", 9, "bold")
            ).grid(
                row=0,
                column=col,
                padx=8,
                pady=2,
                sticky="w"
            )

        for row, ch in enumerate(
            range(1, 5),
            start=1
        ):

            vars_ = self.channel_vars[ch]

            ttk.Checkbutton(
                channels,
                variable=vars_["enabled"]
            ).grid(
                row=row,
                column=0,
                padx=8
            )

            ttk.Label(
                channels,
                text=f"CH{ch}"
            ).grid(
                row=row,
                column=1,
                padx=8,
                sticky="w"
            )

            ttk.Entry(
                channels,
                textvariable=vars_["name"],
                width=25
            ).grid(
                row=row,
                column=2,
                padx=8,
                sticky="ew"
            )

            ttk.Combobox(
                channels,
                textvariable=vars_["range"],
                values=[
                    str(x)
                    for x in SUPPORTED_RANGES
                ],
                width=10,
                state="readonly"
            ).grid(
                row=row,
                column=3,
                padx=8
            )

            ttk.Combobox(
                channels,
                textvariable=vars_["coupling"],
                values=list(
                    SUPPORTED_COUPLING
                ),
                width=10,
                state="readonly"
            ).grid(
                row=row,
                column=4,
                padx=8
            )

            ttk.Combobox(
                channels,
                textvariable=vars_["filter"],
                values=list(
                    SUPPORTED_FILTERS
                ),
                width=15,
                state="readonly"
            ).grid(
                row=row,
                column=5,
                padx=8
            )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = ttk.LabelFrame(
            root,
            text="Shot Metadata",
            padding=6
        )

        metadata.grid(
            row=3,
            column=0,
            sticky="ew",
            padx=15,
            pady=(0, 5)
        )

        self._add_labeled_entry(
            metadata,
            "Shot #",
            self.shot_number_var,
            0,
            0
        )

        self._add_labeled_entry(
            metadata,
            "Operator",
            self.operator_var,
            0,
            2
        )

        self._add_labeled_entry(
            metadata,
            "Coil",
            self.coil_var,
            0,
            4
        )

        self._add_labeled_entry(
            metadata,
            "Target field",
            self.target_field_var,
            0,
            6
        )

        self._add_labeled_entry(
            metadata,
            "Bank voltage",
            self.bank_voltage_var,
            1,
            0
        )

        self._add_labeled_entry(
            metadata,
            "Notes",
            self.notes_var,
            1,
            2,
            width=45,
            colspan=7
        )

        # ----------------------------------------------------
        # Main bottom area
        # ----------------------------------------------------

        bottom = ttk.Frame(
            root,
            padding=(10, 0, 10, 8)
        )

        bottom.grid(
            row=4,
            column=0,
            sticky="nsew"
        )

        bottom.columnconfigure(
            1,
            weight=1
        )

        bottom.rowconfigure(
            0,
            weight=1
        )

        # Left controls
        controls = ttk.LabelFrame(
            bottom,
            text="Acquisition",
            padding=8
        )

        controls.grid(
            row=0,
            column=0,
            sticky="nsw",
            padx=(0, 12)
        )

        ttk.Button(
            controls,
            text="TEST CONNECTION",
            command=self.test_connection,
            width=25
        ).pack(
            pady=5
        )

        self.arm_button = ttk.Button(
            controls,
            text="ARM DIGITIZER",
            command=self.start_acquisition,
            width=25
        )

        self.arm_button.pack(
            pady=(15, 5),
            ipady=15
        )

        ttk.Button(
            controls,
            text="ABORT",
            command=self.abort_acquisition,
            width=25
        ).pack(
            pady=5
        )

        ttk.Separator(
            controls
        ).pack(
            fill="x",
            pady=15
        )

        ttk.Button(
            controls,
            text="VIEW LAST SHOT",
            command=self.view_last_shot,
            width=25
        ).pack(
            pady=5
        )

        ttk.Button(
            controls,
            text="OPEN DATA FOLDER",
            command=self.open_data_folder,
            width=25
        ).pack(
            pady=5
        )

        ttk.Button(
            controls,
            text="ADVANCED",
            command=self.open_advanced,
            width=25
        ).pack(
            pady=5
        )

        # Transfer progress
        ttk.Label(
            controls,
            text="Transfer:"
        ).pack(
            pady=(15, 2)
        )

        ttk.Label(
            controls,
            textvariable=self.progress_var
        ).pack()

        # ----------------------------------------------------
        # Embedded waveform plot + compact acquisition log
        # ----------------------------------------------------

        display = ttk.Frame(bottom)
        display.grid(row=0, column=1, sticky="nsew")
        display.columnconfigure(0, weight=1)
        display.rowconfigure(0, weight=4)
        display.rowconfigure(1, weight=1)

        plot_frame = ttk.LabelFrame(
            display,
            text="Waveforms — last acquired shot",
            padding=4,
        )
        plot_frame.grid(
            row=0, column=0, sticky="nsew", pady=(0, 5)
        )
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.plot_placeholder = ttk.Label(
            plot_frame,
            text="Waveforms will appear here after a shot is acquired.\\n"
                 "Use VIEW LAST SHOT to redraw the latest acquisition.",
            anchor="center",
            justify="center",
        )
        self.plot_placeholder.grid(row=0, column=0, sticky="nsew")
        self.plot_canvas = None
        self.plot_toolbar = None
        self.plot_figure = None

        console_frame = ttk.LabelFrame(
            display, text="Acquisition Log", padding=5
        )
        console_frame.grid(row=1, column=0, sticky="nsew")
        console_frame.rowconfigure(0, weight=1)
        console_frame.columnconfigure(0, weight=1)

        self.console = tk.Text(
            console_frame, wrap="word", height=5, font=("Consolas", 9)
        )
        self.console.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            console_frame, orient="vertical", command=self.console.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.console.configure(yscrollcommand=scrollbar.set)

        self.log(
            "Ready."
        )

        self.log(
            "MAGNET SHOT mode: "
            "ARM DIGITIZER → wait for external master trigger."
        )

    # ========================================================
    # Helpers
    # ========================================================

    def _add_labeled_entry(
        self,
        parent,
        label,
        variable,
        row,
        column,
        combo=False,
        values=None,
        callback=None,
        width=15,
        colspan=1,
    ):

        ttk.Label(
            parent,
            text=label
        ).grid(
            row=row,
            column=column,
            padx=5,
            pady=2,
            sticky="w"
        )

        if combo:

            widget = ttk.Combobox(
                parent,
                textvariable=variable,
                values=values or [],
                state="readonly",
                width=width
            )

            widget.grid(
                row=row,
                column=column + 1,
                columnspan=colspan,
                padx=5,
                pady=2,
                sticky="ew"
            )

            if callback:
                widget.bind(
                    "<<ComboboxSelected>>",
                    lambda event:
                    callback()
                )

        else:

            widget = ttk.Entry(
                parent,
                textvariable=variable,
                width=width
            )

            widget.grid(
                row=row,
                column=column + 1,
                columnspan=colspan,
                padx=5,
                pady=2,
                sticky="ew"
            )

            widget.bind(
                "<KeyRelease>",
                lambda event:
                self._update_preview()
            )

    # ========================================================
    # Logging
    # ========================================================

    def log(self, text):

        timestamp = time.strftime(
            "%H:%M:%S"
        )

        self.console.insert(
            "end",
            f"[{timestamp}] {text}\n"
        )

        self.console.see(
            "end"
        )

    # ========================================================
    # Mode / trigger changes
    # ========================================================

    def _mode_changed(self):

        mode = self.mode_var.get()

        if mode == "MAGNET SHOT":

            self.trigger_var.set(
                "EXT"
            )

            self.arm_button.configure(
                text="ARM DIGITIZER"
            )

        else:

            self.trigger_var.set(
                "IMM"
            )

            self.arm_button.configure(
                text="ARM / BENCH ACQUIRE"
            )

        self._update_preview()

    def _trigger_changed(self):

        self._update_preview()

    # ========================================================
    # Preview
    # ========================================================

    def _update_preview(self):

        try:

            timing = TimingConfig(
                duration_ms=float(
                    self.duration_var.get()
                ),
                pre_trigger_pct=float(
                    self.pretrigger_var.get()
                )
            )

            text = (
                f"{timing.total_samples:,} samples | "
                f"pre-trigger: "
                f"{timing.pre_trigger_samples:,} | "
                f"window: "
                f"{timing.start_time_ms:.3f} to "
                f"{timing.end_time_ms:.3f} ms | "
                f"Δt = 50 ns"
            )

            self.preview_var.set(
                text
            )

        except Exception:

            self.preview_var.set(
                "Enter valid timing values."
            )

    # ========================================================
    # Build shot configuration
    # ========================================================

    def build_configuration(self):

        mode = self.mode_var.get()

        timing = TimingConfig(
            duration_ms=float(
                self.duration_var.get()
            ),
            pre_trigger_pct=float(
                self.pretrigger_var.get()
            ),
        )

        trigger = TriggerConfig(
            source=self.trigger_var.get(),
            slope=self.slope_var.get(),
            timeout_s=float(
                self.timeout_var.get()
            )
            if self.timeout_var.get()
            else 180.0,
        )

        channels = []

        for ch in range(1, 5):

            vars_ = self.channel_vars[ch]

            channel = ChannelConfig(
                number=ch,
                name=vars_["name"].get(),
                range_v=float(
                    vars_["range"].get()
                ),
                coupling=vars_[
                    "coupling"
                ].get(),
                filter_name=vars_[
                    "filter"
                ].get(),
                enabled=vars_[
                    "enabled"
                ].get(),
            )

            channels.append(
                channel
            )

        metadata = ShotMetadata(
            shot_number=int(
                self.shot_number_var.get()
            ),
            operator=self.operator_var.get(),
            coil=self.coil_var.get(),
            target_field=self.target_field_var.get(),
            bank_voltage=self.bank_voltage_var.get(),
            notes=self.notes_var.get(),
            trigger_description=(
                "External master experiment trigger"
                if trigger.source == "EXT"
                else
                "Software immediate trigger"
            ),
        )

        config = ShotConfiguration(
            mode=mode,
            timing=timing,
            trigger=trigger,
            channels=channels,
            metadata=metadata,
            save_npz=self.npz_var.get(),
            save_hdf5=self.hdf5_var.get(),
            save_csv=self.csv_var.get(),
        )

        config.validate()

        return config

    # ========================================================
    # Connection test
    # ========================================================

    def test_connection(self):

        try:

            if self.driver is not None:
                self.driver.close()

            self.driver = L4534ADriver(
                visa_address=self.visa_var.get(),
                chunk_size=int(
                    self.chunk_var.get()
                ),
                reset_instrument=False,
            )

            identity = self.driver.connect()

            self.status_var.set(
                "CONNECTED"
            )

            self.log(
                f"Connected: {identity}"
            )

        except Exception as exc:

            self.status_var.set(
                "DISCONNECTED"
            )

            messagebox.showerror(
                "Connection error",
                str(exc)
            )

            self.log(
                f"Connection error: {exc}"
            )

    # ========================================================
    # Start acquisition
    # ========================================================

    def start_acquisition(self):

        if (
            self.worker is not None
            and self.worker.is_alive()
        ):
            return

        try:

            config = self.build_configuration()

        except Exception as exc:

            messagebox.showerror(
                "Invalid configuration",
                str(exc)
            )

            return

        if self.driver is None:

            messagebox.showwarning(
                "Digitizer not connected",
                "Connect to the L4534A first."
            )

            return

        self.abort_requested = False

        self.arm_button.configure(
            state="disabled"
        )

        self.status_var.set(
            "ARMING"
        )

        self.progress_var.set(
            "0 / 4"
        )

        self.log(
            f"Preparing Shot "
            f"{config.metadata.shot_number}..."
        )

        if config.mode == "MAGNET SHOT":

            self.log(
                "IMPORTANT: Python will NOT trigger "
                "the capacitor bank."
            )

            self.log(
                "Arm the digitizer first; "
                "then the external master trigger "
                "will start the shot."
            )

        self.worker = threading.Thread(
            target=self._acquisition_worker,
            args=(config,),
            daemon=True,
        )

        self.worker.start()

    # ========================================================
    # Worker
    # ========================================================

    def _acquisition_worker(
        self,
        config
    ):

        try:

            self.events.put(
                (
                    "log",
                    "Configuring L4534A..."
                )
            )

            # Driver status callback.
            def status_callback(message):

                self.events.put(
                    (
                        "status",
                        message
                    )
                )

            def progress_callback(done, total):

                self.events.put(
                    (
                        "progress",
                        done,
                        total
                    )
                )

            self.driver.configure(
                config
            )

            self.driver.verify_configuration(
                config
            )

            self.events.put(
                (
                    "log",
                    "Configuration verified."
                )
            )

            # ------------------------------------------------
            # ARM + INIT
            # ------------------------------------------------

            self.driver.arm(
                config,
                status_callback=status_callback
            )

            self.events.put(
                (
                    "armed",
                    config.mode
                )
            )

            # ------------------------------------------------
            # Wait for hardware/software trigger
            # ------------------------------------------------

            self.driver.wait_for_completion(
                config.trigger.timeout_s,
                status_callback=status_callback
            )

            self.events.put(
                (
                    "status",
                    "ACQUISITION COMPLETE"
                )
            )

            # ------------------------------------------------
            # Transfer
            # ------------------------------------------------

            self.events.put(
                (
                    "status",
                    "TRANSFERRING DATA"
                )
            )

            data = self.driver.fetch_all_channels(
                config.timing.total_samples,
                progress_callback=
                progress_callback,
            )

            self.events.put(
                (
                    "status",
                    "SAVING DATA"
                )
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            result = self.storage.save_shot(
                config,
                data
            )

            self.events.put(
                (
                    "complete",
                    result,
                    data,
                    config
                )
            )

        except Exception as exc:

            self.events.put(
                (
                    "error",
                    str(exc)
                )
            )

    # ========================================================
    # Abort
    # ========================================================

    def abort_acquisition(self):

        if (
            self.worker is None
            or not self.worker.is_alive()
        ):
            return

        self.abort_requested = True

        self.log(
            "Abort requested."
        )

        self.status_var.set(
            "ABORTING"
        )

        try:

            if self.driver is not None:
                self.driver.abort()

        except Exception as exc:

            self.log(
                f"Abort error: {exc}"
            )

    # ========================================================
    # Event processor
    # ========================================================

    def _process_events(self):

        try:

            while True:

                event = self.events.get_nowait()

                event_type = event[0]

                if event_type == "log":

                    self.log(
                        event[1]
                    )

                elif event_type == "status":

                    message = event[1]

                    self.status_var.set(
                        message
                    )

                    self.log(
                        message
                    )

                elif event_type == "armed":

                    mode = event[1]

                    if mode == "MAGNET SHOT":

                        self.status_var.set(
                            "ARMED — WAITING FOR MAGNET TRIGGER"
                        )

                        self.log(
                            "DIGITIZER ARMED."
                        )

                        self.log(
                            "WAITING FOR EXTERNAL MASTER TRIGGER."
                        )

                    else:

                        self.status_var.set(
                            "BENCH TEST — ACQUIRING"
                        )

                elif event_type == "progress":

                    done = event[1]
                    total = event[2]

                    self.progress_var.set(
                        f"{done} / {total}"
                    )

                elif event_type == "complete":

                    result = event[1]
                    data = event[2]
                    config = event[3]

                    self.last_data = data

                    self.last_time = result[
                        "time_ms"
                    ]

                    self.current_shot_folder = (
                        result["folder"]
                    )

                    self.status_var.set(
                        "SHOT COMPLETE"
                    )

                    self.log(
                        "Shot acquisition complete."
                    )

                    self.log(
                        "Saved to:"
                    )

                    self.log(
                        str(
                            result["folder"]
                        )
                    )

                    self.log(
                        "Files:"
                    )

                    for path in result[
                        "files"
                    ]:

                        self.log(
                            f"  {path}"
                        )

                    self._show_diagnostics(
                        data,
                        config
                    )

                    # Automatically increment next shot.
                    self.shot_number_var.set(
                        str(
                            config.metadata.shot_number
                            + 1
                        )
                    )

                    self.arm_button.configure(
                        state="normal"
                    )

                    # Automatically display result.
                    self._show_last_plot(
                        config
                    )

                elif event_type == "error":

                    error = event[1]

                    self.status_var.set(
                        "ERROR"
                    )

                    self.log(
                        f"ERROR: {error}"
                    )

                    self.arm_button.configure(
                        state="normal"
                    )

                    messagebox.showerror(
                        "Acquisition error",
                        error
                    )

        except queue.Empty:
            pass

        self.root.after(
            100,
            self._process_events
        )

    # ========================================================
    # Diagnostics
    # ========================================================

    def _show_diagnostics(
        self,
        data,
        config
    ):

        channel_map = {
            ch.number: ch
            for ch in config.channels
        }

        diagnostics = (
            self.plotter.diagnostics(
                data,
                channel_map
            )
        )

        for channel, values in (
            diagnostics.items()
        ):

            self.log(
                f"CH{channel}: "
                f"min={values['min']:.5g}, "
                f"max={values['max']:.5g}, "
                f"RMS={values['rms']:.5g}"
            )

            if values["saturation"]:

                self.log(
                    f"WARNING CH{channel}: "
                    f"possible saturation."
                )

            if values["flatline"]:

                self.log(
                    f"WARNING CH{channel}: "
                    f"possible flatline."
                )

    # ========================================================
    # Plot
    # ========================================================

    def _show_last_plot(
        self,
        config
    ):

        if (
            self.last_data is None
            or self.last_time is None
        ):
            return

        channel_map = {
            ch.number: ch
            for ch in config.channels
        }

        # Embed the four-channel Matplotlib figure in the main GUI.
        fig = self.plotter.plot(
            self.last_time,
            self.last_data,
            channel_map,
            config.metadata.shot_number
        )

        if self.plot_canvas is not None:
            self.plot_canvas.get_tk_widget().destroy()
            self.plot_canvas = None
        if self.plot_toolbar is not None:
            self.plot_toolbar.destroy()
            self.plot_toolbar = None
        if self.plot_figure is not None:
            import matplotlib.pyplot as plt
            plt.close(self.plot_figure)

        self.plot_placeholder.grid_remove()
        # Fit the figure to the available panel so the four traces remain
        # visible without forcing the main window to grow or scroll.
        self.root.update_idletasks()
        panel = self.plot_placeholder.master
        panel_width = max(700, panel.winfo_width() - 20)
        panel_height = max(300, panel.winfo_height() - 55)
        fig.set_size_inches(
            panel_width / fig.dpi,
            panel_height / fig.dpi,
            forward=True,
        )
        self.plot_figure = fig
        self.plot_canvas = FigureCanvasTkAgg(fig, master=panel)
        self.plot_canvas.get_tk_widget().grid(
            row=0, column=0, sticky="nsew"
        )
        self.plot_toolbar = NavigationToolbar2Tk(
            self.plot_canvas,
            self.plot_placeholder.master,
            pack_toolbar=False,
        )
        self.plot_toolbar.update()
        self.plot_toolbar.grid(row=1, column=0, sticky="ew")
        self.plot_canvas.draw_idle()

    def view_last_shot(self):

        if (
            self.last_data is None
            or self.last_time is None
        ):

            messagebox.showinfo(
                "No shot",
                "No shot has been acquired in this session."
            )

            return

        try:

            config = self.build_configuration()

            self._show_last_plot(
                config
            )

        except Exception as exc:

            messagebox.showerror(
                "Plot error",
                str(exc)
            )

    # ========================================================
    # Data folder
    # ========================================================

    def open_data_folder(self):

        path = Path(
            "data_runs"
        ).resolve()

        path.mkdir(
            parents=True,
            exist_ok=True
        )

        import os

        os.startfile(
            str(path)
        )

    # ========================================================
    # Advanced settings
    # ========================================================

    def open_advanced(self):

        window = tk.Toplevel(
            self.root
        )

        window.title(
            "Advanced / Service Settings"
        )

        window.geometry(
            "700x300"
        )

        frame = ttk.Frame(
            window,
            padding=15
        )

        frame.pack(
            fill="both",
            expand=True
        )

        ttk.Label(
            frame,
            text="VISA address"
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=5
        )

        ttk.Entry(
            frame,
            textvariable=self.visa_var,
            width=65
        ).grid(
            row=0,
            column=1,
            pady=5,
            sticky="ew"
        )

        ttk.Label(
            frame,
            text="VISA chunk size"
        ).grid(
            row=1,
            column=0,
            sticky="w",
            pady=5
        )

        ttk.Entry(
            frame,
            textvariable=self.chunk_var,
            width=20
        ).grid(
            row=1,
            column=1,
            sticky="w",
            pady=5
        )

        ttk.Checkbutton(
            frame,
            text="Reset instrument when connecting",
            variable=self.reset_var
        ).grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="w",
            pady=10
        )

        ttk.Label(
            frame,
            text="Use reset only for service/bench situations. "
                 "It is not required before every magnet shot.",
            wraplength=600
        ).grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="w",
            pady=5
        )

        ttk.Button(
            frame,
            text="Close",
            command=window.destroy
        ).grid(
            row=4,
            column=0,
            columnspan=2,
            pady=20
        )


# ============================================================
# Main
# ============================================================

def main():

    root = tk.Tk()

    app = L4534AGUI(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()